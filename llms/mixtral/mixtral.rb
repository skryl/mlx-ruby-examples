# frozen_string_literal: true

require "json"
require "open3"
require "optparse"
require "pathname"

ROOT = File.expand_path("../..", __dir__)

require "mlx"
require "mlx/dsl"

module MixtralExample
  class ModelArgs
    include MLX::DSL::ConfigSchema

    field :dim, Integer, required: true
    field :n_layers, Integer, required: true
    field :head_dim, Integer, required: true
    field :hidden_dim, Integer, required: true
    field :n_heads, Integer, required: true
    field :n_kv_heads, Integer, required: true
    field :norm_eps, [Integer, Float], required: true
    field :vocab_size, Integer, required: true
    field :moe, Hash, required: true
  end

  class Tokenizer
    SCRIPT_PATH = Pathname.new(__dir__).join("python", "sentencepiece_bridge.py").to_s

    attr_reader :bos_id, :eos_id, :pad_id

    def initialize(model_file:, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @python_bin = python_bin
      @model_file = model_file.to_s
      ids = run_json("ids")
      @bos_id = ids.fetch("bos_id")
      @eos_id = ids.fetch("eos_id")
      @pad_id = ids.fetch("pad_id")
      @sep = "▁"
    end

    def encode(text)
      [@bos_id] + run_json("encode", text.to_s)
    end

    def decode(tokens)
      toks = tokens.map(&:to_i)
      out = run_json("decode", JSON.generate(toks))
      return out if toks.empty?

      first_piece = run_json("piece", toks[0].to_s)
      return " #{out}" if !first_piece.empty? && first_piece[0] == @sep

      out
    end

    private

    def run_json(op, arg = "")
      stdout, stderr, status = Open3.capture3(@python_bin, SCRIPT_PATH, @model_file, op, arg.to_s)
      return JSON.parse(stdout) if status.success?

      raise RuntimeError, "sentencepiece bridge failed: #{stderr}"
    rescue JSON::ParserError => e
      raise RuntimeError, "sentencepiece bridge returned invalid JSON: #{e.message}"
    end
  end

  class Attention < MLX::NN::Module
    def initialize(args)
      super()
      @n_heads = args.n_heads
      @n_kv_heads = args.n_kv_heads
      @head_dim = args.head_dim
      @scale = @head_dim**-0.5

      self.wq = MLX::NN::Linear.new(args.dim, args.n_heads * args.head_dim, bias: false)
      self.wk = MLX::NN::Linear.new(args.dim, args.n_kv_heads * args.head_dim, bias: false)
      self.wv = MLX::NN::Linear.new(args.dim, args.n_kv_heads * args.head_dim, bias: false)
      self.wo = MLX::NN::Linear.new(args.n_heads * args.head_dim, args.dim, bias: false)
      self.rope = MLX::NN::RoPE.new(args.head_dim, traditional: true, base: 1_000_000)
    end

    def call(x, mask: nil, cache: nil)
      batch_size, seq_len, _dims = x.shape
      queries = wq.call(x)
      keys = wk.call(x)
      values = wv.call(x)

      queries = MLX::Core.transpose(
        MLX::Core.reshape(queries, [batch_size, seq_len, @n_heads, @head_dim]),
        [0, 2, 1, 3]
      )
      keys = MLX::Core.transpose(
        MLX::Core.reshape(keys, [batch_size, seq_len, @n_kv_heads, @head_dim]),
        [0, 2, 1, 3]
      )
      values = MLX::Core.transpose(
        MLX::Core.reshape(values, [batch_size, seq_len, @n_kv_heads, @head_dim]),
        [0, 2, 1, 3]
      )

      if !cache.nil?
        key_cache, value_cache = cache
        offset = key_cache.shape[2]
        queries = rope.call(queries, offset: offset)
        keys = rope.call(keys, offset: offset)
        keys = MLX::Core.concatenate([key_cache, keys], 2)
        values = MLX::Core.concatenate([value_cache, values], 2)
      else
        queries = rope.call(queries)
        keys = rope.call(keys)
      end

      attn_mask = mask.nil? ? nil : mask.astype(queries.dtype)
      output = MLX::Core.scaled_dot_product_attention(queries, keys, values, @scale, attn_mask)
      output = MLX::Core.transpose(output, [0, 2, 1, 3])
      output = MLX::Core.reshape(output, [batch_size, seq_len, @n_heads * @head_dim])
      [wo.call(output), [keys, values]]
    end
  end

  class FeedForward < MLX::NN::Module
    def initialize(args)
      super()
      self.w1 = MLX::NN::Linear.new(args.dim, args.hidden_dim, bias: false)
      self.w2 = MLX::NN::Linear.new(args.hidden_dim, args.dim, bias: false)
      self.w3 = MLX::NN::Linear.new(args.dim, args.hidden_dim, bias: false)
    end

    def call(x)
      w2.call(MLX::Core.multiply(MLX::NN.silu(w1.call(x)), w3.call(x)))
    end
  end

  class MOEFeedForward < MLX::NN::Module
    def initialize(args)
      super()
      @num_experts = args.moe.fetch("num_experts")
      @num_experts_per_tok = args.moe.fetch("num_experts_per_tok")
      self.experts = Array.new(@num_experts) { FeedForward.new(args) }
      self.gate = MLX::NN::Linear.new(args.dim, @num_experts, bias: false)
    end

    def call(x)
      ne = @num_experts_per_tok
      orig_shape = x.shape
      dims = x.shape[-1]
      tokens = x.size / dims
      x = MLX::Core.reshape(x, [tokens, dims])

      gates = gate.call(x)
      inds = MLX::Core.argpartition(MLX::Core.multiply(gates, -1.0), ne - 1, -1)
      take_ids = MLX::Core.arange(0, ne, 1, MLX::Core.int32)
      inds = MLX::Core.take(inds, take_ids, 1)

      scores = MLX::Core.take_along_axis(gates, inds, -1)
      scores = MLX::Core.softmax(scores.astype(MLX::Core.float32), -1).astype(gates.dtype)

      inds_list = inds.to_a
      rows = []
      (0...x.shape[0]).each do |i|
        xt = x[i]
        st = scores[i]
        selected = inds_list[i]
        yt = MLX::Core.concatenate(
          selected.map { |expert_idx| MLX::Core.expand_dims(experts[expert_idx].call(xt), 1) },
          1
        )
        yt = MLX::Core.sum(MLX::Core.multiply(yt, st), -1)
        rows << MLX::Core.expand_dims(yt, 0)
      end

      y = MLX::Core.concatenate(rows, 0)
      MLX::Core.reshape(y, orig_shape)
    end
  end

  class MOETransformerBlock < MLX::NN::Module
    def initialize(args)
      super()
      self.attention = Attention.new(args)
      self.feed_forward = MOEFeedForward.new(args)
      self.attention_norm = MLX::NN::RMSNorm.new(args.dim, eps: args.norm_eps)
      self.ffn_norm = MLX::NN::RMSNorm.new(args.dim, eps: args.norm_eps)
    end

    def call(x, mask: nil, cache: nil)
      residual, next_cache = attention.call(attention_norm.call(x), mask: mask, cache: cache)
      hidden = MLX::Core.add(x, residual)
      residual = feed_forward.call(ffn_norm.call(hidden))
      [MLX::Core.add(hidden, residual), next_cache]
    end
  end

  class Mixtral < MLX::NN::Module
    attr_reader :vocab_size, :n_layers

    def initialize(args)
      super()
      @vocab_size = args.vocab_size
      @n_layers = args.n_layers
      if @vocab_size <= 0
        raise ArgumentError, "vocab_size must be positive"
      end

      self.tok_embeddings = MLX::NN::Embedding.new(args.vocab_size, args.dim)
      self.layers = Array.new(args.n_layers) { MOETransformerBlock.new(args) }
      self.norm = MLX::NN::RMSNorm.new(args.dim, eps: args.norm_eps)
      self.output = MLX::NN::Linear.new(args.dim, args.vocab_size, bias: false)
    end

    def call(inputs, cache: nil)
      hidden = tok_embeddings.call(inputs)
      seq_len = hidden.shape[1]
      offset = MLX::DSL::Positions.offset_from_cache(cache, layer: 0)

      mask = nil
      if seq_len > 1 || offset.positive?
        mask = MLX::DSL::Masks.causal(length: seq_len, offset: offset, dtype: hidden.dtype)
      end

      cache_state = cache || Array.new(layers.length)
      hidden, next_cache = MLX::DSL.run_stack(layers, hidden, mask: mask, cache: cache_state)

      hidden = norm.call(hidden)
      last_idx = MLX::Core.array([seq_len - 1], MLX::Core.int32)
      last_hidden = MLX::Core.take(hidden, last_idx, 1)
      [output.call(last_hidden), next_cache]
    end
  end

  module_function

  def load_weights(model_path)
    files = if File.file?(model_path.join("weights.npz"))
      [model_path.join("weights.npz").to_s]
    else
      Dir.glob(model_path.join("weights.*.npz").to_s).sort
    end
    raise Errno::ENOENT, "No weight files found in #{model_path}" if files.empty?

    files.each_with_object({}) do |wf, out|
      MLX::Core.load(wf).to_a.each do |key, value|
        out[key.to_s] = value
      end
    end
  end

  def load_model(folder, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    model_path = Pathname.new(folder.to_s)
    tokenizer = Tokenizer.new(model_file: model_path.join("tokenizer.model"), python_bin: python_bin)
    config = JSON.parse(File.binread(model_path.join("config.json")))
    config.delete("model_type")
    quantization = config.delete("quantization")
    model_args = ModelArgs.from_hash(
      dim: config.fetch("dim"),
      n_layers: config.fetch("n_layers"),
      head_dim: config.fetch("head_dim"),
      hidden_dim: config.fetch("hidden_dim"),
      n_heads: config.fetch("n_heads"),
      n_kv_heads: config.fetch("n_kv_heads"),
      norm_eps: config.fetch("norm_eps"),
      vocab_size: config.fetch("vocab_size"),
      moe: config.fetch("moe")
    )

    model = Mixtral.new(model_args)
    if !quantization.nil?
      MLX::NN.quantize(model, **quantization.transform_keys(&:to_sym))
    end
    model.update(MLX::Utils.tree_unflatten(load_weights(model_path).to_a))
    [model, tokenizer]
  end

  def generate(prompt, model, temp: 0.0, max_tokens: 1_000_000)
    sampler = if temp.to_f.zero?
      { strategy: :argmax }
    else
      { strategy: :temperature, temperature: temp.to_f }
    end
    generator = MLX::DSL::Generate.new(model: model, sampler: sampler, mode: :decoder_only)
    Enumerator.new do |emitter|
      generator.each_token(input_ids: MLX::Core.expand_dims(prompt, 0), max_tokens: max_tokens) do |token_id, _chunk|
        emitter << MLX::Core.array(token_id, MLX::Core.int32)
      end
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model_path: "mlx_model",
    prompt: "In the beginning the Universe was created.",
    max_tokens: 100,
    temp: 0.0,
    seed: 0,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/mixtral/mixtral.rb [options]"
    opts.on("--model-path PATH", String, "Path to model weights, tokenizer, config") { |v| options[:model_path] = v }
    opts.on("--prompt TEXT", String, "Prompt text") { |v| options[:prompt] = v }
    opts.on("--max-tokens N", Integer, "Maximum number of generated tokens") { |v| options[:max_tokens] = v }
    opts.on("--temp N", Float, "Sampling temperature") { |v| options[:temp] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--python-bin BIN", String, "Python binary for sentencepiece bridge") { |v| options[:python_bin] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])
  puts "[INFO] Loading model from disk."
  model, tokenizer = MixtralExample.load_model(options[:model_path], python_bin: options[:python_bin])
  puts "[INFO] Starting generation..."

  print options[:prompt]
  prompt = MLX::Core.array(tokenizer.encode(options[:prompt]), MLX::Core.int32)
  tokens = []
  stop = false

  MixtralExample.generate(prompt, model, temp: options[:temp], max_tokens: options[:max_tokens]).each_with_index do |token, idx|
    tokens << token
    if (tokens.length % 10).zero?
      MLX::Core.eval(*tokens)
      eos_index = tokens.find_index { |t| t.item.to_i == tokenizer.eos_id }
      chunk = eos_index.nil? ? tokens : tokens.take(eos_index)
      text = tokenizer.decode(chunk.map { |t| t.item.to_i })
      print(text, end: "", flush: true)
      tokens = []
      if !eos_index.nil?
        stop = true
        break
      end
    end
    break if idx + 1 >= options[:max_tokens]
  end

  unless stop
    MLX::Core.eval(*tokens) unless tokens.empty?
    text = tokenizer.decode(tokens.map { |t| t.item.to_i })
    print(text, flush: true)
  end
  puts
end
