# frozen_string_literal: true

require "json"
require "open3"
require "optparse"
require "pathname"
require "time"

ROOT = File.expand_path("../..", __dir__)

require "mlx"

module MistralExample
  class ModelArgs
    attr_reader :dim, :n_layers, :head_dim, :hidden_dim, :n_heads, :n_kv_heads, :norm_eps, :vocab_size, :rope_theta

    def initialize(
      dim:,
      n_layers:,
      head_dim:,
      hidden_dim:,
      n_heads:,
      n_kv_heads:,
      norm_eps:,
      vocab_size:,
      rope_theta: 10_000
    )
      @dim = dim
      @n_layers = n_layers
      @head_dim = head_dim
      @hidden_dim = hidden_dim
      @n_heads = n_heads
      @n_kv_heads = n_kv_heads
      @norm_eps = norm_eps
      @vocab_size = vocab_size
      @rope_theta = rope_theta
    end
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
      self.rope = MLX::NN::RoPE.new(args.head_dim, traditional: true, base: args.rope_theta)
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

      output = MLX::Core.scaled_dot_product_attention(queries, keys, values, @scale, mask)
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

  class TransformerBlock < MLX::NN::Module
    def initialize(args)
      super()
      self.attention = Attention.new(args)
      self.feed_forward = FeedForward.new(args)
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

  class Mistral < MLX::NN::Module
    attr_reader :vocab_size, :n_layers

    def initialize(args)
      super()
      @vocab_size = args.vocab_size
      @n_layers = args.n_layers
      if @vocab_size <= 0
        raise ArgumentError, "vocab_size must be positive"
      end

      self.tok_embeddings = MLX::NN::Embedding.new(args.vocab_size, args.dim)
      self.layers = Array.new(args.n_layers) { TransformerBlock.new(args) }
      self.norm = MLX::NN::RMSNorm.new(args.dim, eps: args.norm_eps)
      self.output = MLX::NN::Linear.new(args.dim, args.vocab_size, bias: false)
    end

    def call(inputs, cache: nil)
      hidden = tok_embeddings.call(inputs)

      mask = nil
      if hidden.shape[1] > 1
        mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(hidden.shape[1])
        mask = mask.astype(hidden.dtype)
      end

      cache ||= Array.new(layers.length)
      layers.each_with_index do |layer, i|
        hidden, cache[i] = layer.call(hidden, mask: mask, cache: cache[i])
      end

      [output.call(norm.call(hidden)), cache]
    end
  end

  module_function

  def load_weights(model_dir)
    unsharded = model_dir.join("weights.npz")
    files = if File.file?(unsharded)
      [unsharded.to_s]
    else
      shard_files = Dir.glob(model_dir.join("weights.*.npz").to_s).sort
      raise Errno::ENOENT, "No weights found in #{model_dir}" if shard_files.empty?

      shard_files
    end

    files.each_with_object({}) do |file, out|
      MLX::Core.load(file).to_a.each do |key, value|
        out[key.to_s] = value
      end
    end
  end

  def load_model(folder, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    model_path = Pathname.new(folder.to_s)
    tokenizer = Tokenizer.new(model_file: model_path.join("tokenizer.model"), python_bin: python_bin)

    config = JSON.parse(File.binread(model_path.join("config.json")))
    config.delete("sliding_window")
    config.delete("model_type")
    quantization = config.delete("quantization")
    config["rope_theta"] = 10_000 unless config.key?("rope_theta")

    args = ModelArgs.new(
      dim: config.fetch("dim"),
      n_layers: config.fetch("n_layers"),
      head_dim: config.fetch("head_dim"),
      hidden_dim: config.fetch("hidden_dim"),
      n_heads: config.fetch("n_heads"),
      n_kv_heads: config.fetch("n_kv_heads"),
      norm_eps: config.fetch("norm_eps"),
      vocab_size: config.fetch("vocab_size"),
      rope_theta: config.fetch("rope_theta")
    )
    model = Mistral.new(args)
    if !quantization.nil?
      MLX::NN.quantize(model, **quantization.transform_keys(&:to_sym))
    end
    model.update(MLX::Utils.tree_unflatten(load_weights(model_path).to_a))
    MLX::Core.eval(model.parameters)
    [model, tokenizer]
  end

  def generate(prompt, model, temp: 0.0)
    sample = lambda do |logits|
      if temp.to_f.zero?
        MLX::Core.argmax(logits, -1)
      else
        MLX::Core.categorical(MLX::Core.multiply(logits, 1.0 / temp.to_f))
      end
    end

    Enumerator.new do |emitter|
      logits, cache = model.call(MLX::Core.expand_dims(prompt, 0))
      y = sample.call(last_logits(logits))
      emitter << y

      loop do
        logits, cache = model.call(MLX::Core.expand_dims(y, 1), cache: cache)
        y = sample.call(MLX::Core.squeeze(logits, 1))
        emitter << y
      end
    end
  end

  def run_generation(model:, tokenizer:, prompt:, max_tokens:, temp:, tokens_per_eval:)
    print prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = MLX::Core.array(prompt_tokens, MLX::Core.int32)

    tokens = []
    generated_count = 0
    prompt_tps = nil
    start = Time.now

    generate(prompt_array, model, temp: temp).each_with_index do |token, idx|
      generated_count = idx + 1
      tokens << token

      if idx.zero?
        MLX::Core.eval(*tokens)
        elapsed = Time.now - start
        prompt_tps = prompt_array.size / [elapsed, 1e-9].max
        start = Time.now
      end

      if (tokens.length % tokens_per_eval).zero?
        MLX::Core.eval(*tokens)
        text = tokenizer.decode(tokens.map { |t| t.item.to_i })
        print(text, end: "", flush: true)
        tokens = []
      end

      break if generated_count >= max_tokens
    end

    unless tokens.empty?
      MLX::Core.eval(*tokens)
      text = tokenizer.decode(tokens.map { |t| t.item.to_i })
      print(text, flush: true)
    end
    puts
    puts "------"
    if !prompt_tps.nil?
      puts format("Tokens per second: prompt %.3f", prompt_tps)
    end
    generation_tps = generated_count / [Time.now - start, 1e-9].max
    puts format("Tokens per second: generation %.3f", generation_tps)
  end

  def last_logits(logits)
    index = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
    token = MLX::Core.take(logits, index, 1)
    MLX::Core.squeeze(token, 1)
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model_path: "mlx_model",
    prompt: "In the beginning the Universe was created.",
    max_tokens: 100,
    temp: 0.0,
    tokens_per_eval: 10,
    seed: 0,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/mistral/mistral.rb [options]"
    opts.on("--model-path PATH", String, "Path to model weights + tokenizer") { |v| options[:model_path] = v }
    opts.on("--prompt TEXT", String, "Prompt text") { |v| options[:prompt] = v }
    opts.on("--max-tokens N", Integer, "Maximum number of generated tokens") { |v| options[:max_tokens] = v }
    opts.on("--temp N", Float, "Sampling temperature") { |v| options[:temp] = v }
    opts.on("--tokens-per-eval N", Integer, "Decode/flush frequency") { |v| options[:tokens_per_eval] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--python-bin BIN", String, "Python binary for sentencepiece bridge") { |v| options[:python_bin] = v }
  end
  parser.parse!

  if options[:tokens_per_eval] <= 0
    raise ArgumentError, "--tokens-per-eval must be positive"
  end

  MLX::Core.random_seed(options[:seed])
  puts "[INFO] Loading model from disk."
  model, tokenizer = MistralExample.load_model(options[:model_path], python_bin: options[:python_bin])
  puts "[INFO] Starting generation..."
  MistralExample.run_generation(
    model: model,
    tokenizer: tokenizer,
    prompt: options[:prompt],
    max_tokens: options[:max_tokens],
    temp: options[:temp],
    tokens_per_eval: options[:tokens_per_eval]
  )
end
