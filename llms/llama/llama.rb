# frozen_string_literal: true

require "json"
require "open3"
require "optparse"
require "pathname"
require "time"

ROOT = File.expand_path("../..", __dir__)

require "mlx"

module LlamaExample
  class ModelArgs
    attr_reader :dim, :n_layers, :head_dim, :hidden_dim, :n_heads, :n_kv_heads, :norm_eps, :vocab_size, :rope_theta,
                :rope_traditional

    def initialize(
      dim:,
      n_layers:,
      head_dim:,
      hidden_dim:,
      n_heads:,
      n_kv_heads:,
      norm_eps:,
      vocab_size:,
      rope_theta:,
      rope_traditional: true
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
      @rope_traditional = rope_traditional
    end
  end

  class SentencePieceTokenizer
    SCRIPT_PATH = Pathname.new(__dir__).join("python", "sentencepiece_bridge.py").to_s

    attr_reader :bos_id, :eos_id, :pad_id

    def initialize(model_file:, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @python_bin = python_bin
      @model_file = model_file.to_s
      ids = run_json("ids")
      @bos_id = ids.fetch("bos_id")
      @eos_id = ids.fetch("eos_id")
      @pad_id = ids.fetch("pad_id")
    end

    def encode(text)
      run_json("encode", text.to_s)
    end

    def decode(tokens)
      run_json("decode", JSON.generate(tokens))
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
      @repeats = @n_heads / @n_kv_heads
      @scale = args.head_dim**-0.5

      self.wq = MLX::NN::Linear.new(args.dim, args.n_heads * args.head_dim, bias: false)
      self.wk = MLX::NN::Linear.new(args.dim, args.n_kv_heads * args.head_dim, bias: false)
      self.wv = MLX::NN::Linear.new(args.dim, args.n_kv_heads * args.head_dim, bias: false)
      self.wo = MLX::NN::Linear.new(args.n_heads * args.head_dim, args.dim, bias: false)
      self.rope = MLX::NN::RoPE.new(args.head_dim, traditional: args.rope_traditional, base: args.rope_theta)
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

      keys = repeat_heads(keys, batch_size, seq_len)
      values = repeat_heads(values, batch_size, seq_len)

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

      scores = MLX::Core.matmul(
        MLX::Core.multiply(queries, @scale),
        MLX::Core.transpose(keys, [0, 1, 3, 2])
      )
      scores = MLX::Core.add(scores, mask) unless mask.nil?
      scores = MLX::Core.softmax(scores.astype(MLX::Core.float32), -1).astype(scores.dtype)
      output = MLX::Core.matmul(scores, values)
      output = MLX::Core.transpose(output, [0, 2, 1, 3])
      output = MLX::Core.reshape(output, [batch_size, seq_len, @n_heads * @head_dim])
      [wo.call(output), [keys, values]]
    end

    private

    def repeat_heads(array, batch_size, seq_len)
      expanded = MLX::Core.expand_dims(array, 2)
      repeated = MLX::Core.concatenate(Array.new(@repeats, expanded), 2)
      MLX::Core.reshape(repeated, [batch_size, @n_heads, seq_len, @head_dim])
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
      gated = MLX::NN.silu(w1.call(x))
      up = w3.call(x)
      w2.call(MLX::Core.multiply(gated, up))
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

  class Llama < MLX::NN::Module
    def initialize(args)
      super()
      self.vocab_size = args.vocab_size
      self.tok_embeddings = MLX::NN::Embedding.new(args.vocab_size, args.dim)
      self.layers = Array.new(args.n_layers) { TransformerBlock.new(args) }
      self.norm = MLX::NN::RMSNorm.new(args.dim, eps: args.norm_eps)
      self.output = MLX::NN::Linear.new(args.dim, args.vocab_size, bias: false)
    end

    def call(x)
      mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(x.shape[1])
      mask = mask.astype(tok_embeddings.weight.dtype)
      hidden = tok_embeddings.call(x)
      layers.each do |layer|
        hidden, = layer.call(hidden, mask: mask)
      end
      output.call(norm.call(hidden))
    end

    def generate(x, temp: 1.0)
      sample = lambda do |logits|
        if temp.to_f.zero?
          MLX::Core.argmax(logits, -1)
        else
          MLX::Core.categorical(MLX::Core.multiply(logits, 1.0 / temp.to_f))
        end
      end

      Enumerator.new do |emitter|
        cache = []
        mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(tok_embeddings.weight.dtype)
        hidden = tok_embeddings.call(x)
        layers.each do |layer|
          hidden, layer_cache = layer.call(hidden, mask: mask)
          cache << layer_cache
        end
        hidden = norm.call(hidden)
        y = sample.call(output.call(last_token(hidden)))
        emitter << y

        loop do
          hidden = tok_embeddings.call(MLX::Core.expand_dims(y, 1))
          cache.each_index do |i|
            hidden, cache[i] = layers[i].call(hidden, cache: cache[i])
          end
          hidden = norm.call(hidden)
          y = sample.call(output.call(last_token(hidden)))
          emitter << y
        end
      end
    end

    private

    def last_token(x)
      index = MLX::Core.array([x.shape[1] - 1], MLX::Core.int32)
      token = MLX::Core.take(x, index, 1)
      MLX::Core.squeeze(token, 1)
    end
  end

  module_function

  def sanitize_config(config, weights)
    out = config.dup
    out.delete("model_type")
    n_heads = out.fetch("n_heads")
    out["n_kv_heads"] = n_heads unless out.key?("n_kv_heads")
    out["head_dim"] = out.fetch("dim") / n_heads unless out.key?("head_dim")
    unless out.key?("hidden_dim")
      out["hidden_dim"] = weights.fetch("layers.0.feed_forward.w1.weight").shape[0]
    end
    if out.fetch("vocab_size", -1) < 0
      out["vocab_size"] = weights.fetch("output.weight").shape[-1]
    end
    out["rope_theta"] = 10_000 unless out.key?("rope_theta")
    out["rope_traditional"] = true unless out.key?("rope_traditional")
    out.delete("multiple_of")
    out.delete("ffn_dim_multiplier")
    out
  end

  def load_model(model_path, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    model_dir = Pathname.new(model_path.to_s)
    weights = load_weights(model_dir)
    config_path = model_dir.join("config.json")
    config = sanitize_config(JSON.parse(File.binread(config_path)), weights)
    quantization = config.delete("quantization")

    args = ModelArgs.new(
      dim: config.fetch("dim"),
      n_layers: config.fetch("n_layers"),
      head_dim: config.fetch("head_dim"),
      hidden_dim: config.fetch("hidden_dim"),
      n_heads: config.fetch("n_heads"),
      n_kv_heads: config.fetch("n_kv_heads"),
      norm_eps: config.fetch("norm_eps"),
      vocab_size: config.fetch("vocab_size"),
      rope_theta: config.fetch("rope_theta"),
      rope_traditional: config.fetch("rope_traditional")
    )
    model = Llama.new(args)
    if !quantization.nil?
      MLX::NN.quantize(model, **quantization.transform_keys(&:to_sym))
    end
    model.update(MLX::Utils.tree_unflatten(weights.to_a))
    tokenizer = SentencePieceTokenizer.new(model_file: model_dir.join("tokenizer.model"), python_bin: python_bin)
    [model, tokenizer]
  end

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

  def run_generation(model:, tokenizer:, prompt:, max_tokens:, temp:, write_every:)
    puts "------"
    print prompt
    x = MLX::Core.array([[tokenizer.bos_id] + tokenizer.encode(prompt)], MLX::Core.int32)
    generated = []
    skip = 0

    start = Time.now
    prompt_processing = nil
    model.generate(x, temp: temp).each_with_index do |token, idx|
      generated << token
      if idx.zero?
        MLX::Core.eval(token)
        elapsed = Time.now - start
        prompt_processing = format("[INFO] Prompt processing: %.3f s", elapsed)
        start = Time.now
      end

      break if generated.length >= max_tokens
      break if token.item.to_i == tokenizer.eos_id

      next unless (generated.length % write_every).zero?

      MLX::Core.eval(*generated)
      text = tokenizer.decode(generated.map { |t| t.item.to_i })
      chunk = text[skip..]
      print(chunk, end: "", flush: true) unless chunk.nil?
      skip = text.length
    end

    MLX::Core.eval(*generated) unless generated.empty?
    text = tokenizer.decode(generated.map { |t| t.item.to_i })
    chunk = text[skip..]
    print(chunk, flush: true) unless chunk.nil?
    puts
    puts "------"
    puts prompt_processing unless prompt_processing.nil?
    puts format("[INFO] Full generation: %.3f s", Time.now - start)
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model_path: "mlx_model",
    prompt: "In the beginning the Universe was created.",
    max_tokens: 100,
    write_every: 1,
    temp: 0.0,
    seed: 0,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/llama/llama.rb [options]"
    opts.on("--model-path PATH", String, "Path to converted model files") { |v| options[:model_path] = v }
    opts.on("--prompt TEXT", String, "Prompt text") { |v| options[:prompt] = v }
    opts.on("--max-tokens N", Integer, "Maximum tokens to generate") { |v| options[:max_tokens] = v }
    opts.on("--write-every N", Integer, "Decode frequency") { |v| options[:write_every] = v }
    opts.on("--temp N", Float, "Sampling temperature") { |v| options[:temp] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--python-bin BIN", String, "Python binary for sentencepiece bridge") { |v| options[:python_bin] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])
  model, tokenizer = LlamaExample.load_model(options[:model_path], python_bin: options[:python_bin])
  LlamaExample.run_generation(
    model: model,
    tokenizer: tokenizer,
    prompt: options[:prompt],
    max_tokens: options[:max_tokens],
    temp: options[:temp],
    write_every: options[:write_every]
  )
end
