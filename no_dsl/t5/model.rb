# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

ROOT = File.expand_path("..", __dir__)
DSL_LIB = File.join(ROOT, "codex-dsl", "lib")
$LOAD_PATH.unshift(DSL_LIB) unless $LOAD_PATH.include?(DSL_LIB)

require "mlx"

module T5Example
  SCRIPT_DIR = Pathname.new(__dir__).join("python")

  class T5Config
    attr_reader :d_model,
                :d_kv,
                :d_ff,
                :num_heads,
                :num_layers,
                :num_decoder_layers,
                :layer_norm_epsilon,
                :relative_attention_num_buckets,
                :relative_attention_max_distance,
                :feed_forward_proj,
                :tie_word_embeddings,
                :vocab_size,
                :decoder_start_token_id

    def initialize(config)
      @d_model = config.fetch("d_model")
      @d_kv = config.fetch("d_kv")
      @d_ff = config["d_ff"]
      @num_heads = config.fetch("num_heads")
      @num_layers = config.fetch("num_layers")
      @num_decoder_layers = config["num_decoder_layers"] || @num_layers
      @layer_norm_epsilon = config.fetch("layer_norm_epsilon")
      @relative_attention_num_buckets = config.fetch("relative_attention_num_buckets")
      @relative_attention_max_distance = config.fetch("relative_attention_max_distance")
      @feed_forward_proj = config.fetch("feed_forward_proj")
      @tie_word_embeddings = config.fetch("tie_word_embeddings")
      @vocab_size = config.fetch("vocab_size")
      @decoder_start_token_id = config.fetch("decoder_start_token_id")
    end
  end

  class Tokenizer
    SCRIPT_PATH = Pathname.new(__dir__).join("python", "tokenizer_bridge.py").to_s

    attr_reader :eos_id, :decoder_start_id

    def initialize(model_name, decoder_start_id:, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @model_name = model_name.to_s
      @python_bin = python_bin
      ids = run_json("ids")
      @eos_id = ids.fetch("eos_id")
      @decoder_start_id = ids["decoder_start_id"] || decoder_start_id
    end

    def encode(text)
      run_json("encode", text.to_s)
    end

    def decode(tokens, with_sep: true)
      pieces = run_json("tokens", JSON.generate(tokens.map(&:to_i)))
      sep = with_sep ? " " : ""
      pieces.join.gsub("▁", sep)
    end

    def decode_full(tokens, skip_special_tokens: true)
      payload = { "tokens" => tokens.map(&:to_i), "skip_special_tokens" => skip_special_tokens }
      run_json("decode", JSON.generate(payload))
    end

    private

    def run_json(op, arg = "")
      stdout, stderr, status = Open3.capture3(@python_bin, SCRIPT_PATH, @model_name, op.to_s, arg.to_s)
      return JSON.parse(stdout) if status.success?

      raise RuntimeError, "tokenizer bridge failed: #{stderr}"
    rescue JSON::ParserError => e
      raise RuntimeError, "tokenizer bridge returned invalid JSON: #{e.message}"
    end
  end

  module_function

  def load_t5_config(model_name, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    script_path = SCRIPT_DIR.join("load_t5_config.py").to_s
    stdout, stderr, status = Open3.capture3(python_bin, script_path, model_name.to_s)
    unless status.success?
      raise RuntimeError, "Failed to load T5 config for #{model_name}: #{stderr}"
    end

    T5Config.new(JSON.parse(stdout))
  end

  def load_model(
    model_name,
    weights_path: nil,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  )
    config = load_t5_config(model_name, python_bin: python_bin)
    model = Model.new(config)
    resolved_weights = weights_path || "#{model_name.tr('/', '-')}.npz"
    unless File.exist?(resolved_weights)
      raise Errno::ENOENT, "Weights not found at #{resolved_weights}. Run ruby t5/convert.rb --model #{model_name}"
    end
    model.load_weights(resolved_weights)
    MLX::Core.eval(model.parameters)
    tokenizer = Tokenizer.new(model_name, decoder_start_id: config.decoder_start_token_id, python_bin: python_bin)
    [model, tokenizer]
  end

  def relative_position_bucket(
    relative_position,
    bidirectional: true,
    num_buckets: 32,
    max_distance: 128
  )
    buckets = MLX::Core.zeros_like(relative_position).astype(MLX::Core.int16)
    if bidirectional
      half = num_buckets / 2
      buckets = MLX::Core.add(
        buckets,
        MLX::Core.multiply(MLX::Core.greater(relative_position, 0).astype(MLX::Core.int16), half)
      )
      relative_position = MLX::Core.abs(relative_position)
      num_buckets = half
    else
      relative_position = MLX::Core.multiply(
        MLX::Core.minimum(relative_position, MLX::Core.zeros_like(relative_position)),
        -1
      )
    end

    max_exact = num_buckets / 2
    is_small = MLX::Core.less(relative_position, max_exact)
    scale = (num_buckets - max_exact) / Math.log(max_distance.to_f / max_exact)

    relative_position_if_large = MLX::Core.add(
      max_exact,
      MLX::Core.multiply(
        MLX::Core.log(MLX::Core.divide(relative_position.astype(MLX::Core.float32), max_exact)),
        scale
      ).astype(MLX::Core.int16)
    )
    relative_position_if_large = MLX::Core.minimum(relative_position_if_large, num_buckets - 1)

    MLX::Core.add(
      buckets,
      MLX::Core.where(is_small, relative_position, relative_position_if_large).astype(MLX::Core.int16)
    )
  end

  def create_additive_causal_mask(n, offset: 0)
    rinds = MLX::Core.arange(0, offset + n, 1)
    linds = if offset.zero?
      rinds
    else
      MLX::Core.arange(offset, offset + n, 1)
    end
    lhs = MLX::Core.expand_dims(linds, 1)
    rhs = MLX::Core.expand_dims(rinds, 0)
    mask = MLX::Core.less(lhs, rhs).astype(MLX::Core.float32)
    MLX::Core.multiply(mask, -1e9)
  end

  class RelativePositionBias < MLX::NN::Module
    def initialize(config, bidirectional)
      super()
      @bidirectional = bidirectional
      @num_buckets = config.relative_attention_num_buckets
      @max_distance = config.relative_attention_max_distance
      self.embeddings = MLX::NN::Embedding.new(config.relative_attention_num_buckets, config.num_heads)
    end

    def call(query_length, key_length, offset: 0)
      context_position = MLX::Core.expand_dims(MLX::Core.arange(offset, query_length, 1), 1)
      memory_position = MLX::Core.expand_dims(MLX::Core.arange(0, key_length, 1), 0)
      relative_position = MLX::Core.subtract(memory_position, context_position)

      bucket = T5Example.relative_position_bucket(
        relative_position,
        bidirectional: @bidirectional,
        num_buckets: @num_buckets,
        max_distance: @max_distance
      )
      values = embeddings.call(bucket)
      MLX::Core.transpose(values, [2, 0, 1])
    end
  end

  class MultiHeadAttention < MLX::NN::Module
    def initialize(config)
      super()
      inner_dim = config.d_kv * config.num_heads
      @num_heads = config.num_heads
      @head_dim = config.d_kv
      self.query_proj = MLX::NN::Linear.new(config.d_model, inner_dim, bias: false)
      self.key_proj = MLX::NN::Linear.new(config.d_model, inner_dim, bias: false)
      self.value_proj = MLX::NN::Linear.new(config.d_model, inner_dim, bias: false)
      self.out_proj = MLX::NN::Linear.new(inner_dim, config.d_model, bias: false)
    end

    def call(queries, keys, values, mask: nil, cache: nil)
      queries = query_proj.call(queries)
      keys = key_proj.call(keys)
      values = value_proj.call(values)

      batch_size, q_len, _ = queries.shape
      _b, k_len, _ = keys.shape
      queries = MLX::Core.transpose(
        MLX::Core.reshape(queries, [batch_size, q_len, @num_heads, @head_dim]),
        [0, 2, 1, 3]
      )
      keys = MLX::Core.transpose(
        MLX::Core.reshape(keys, [batch_size, k_len, @num_heads, @head_dim]),
        [0, 2, 1, 3]
      )
      values = MLX::Core.transpose(
        MLX::Core.reshape(values, [batch_size, k_len, @num_heads, @head_dim]),
        [0, 2, 1, 3]
      )

      if !cache.nil?
        key_cache, value_cache = cache
        keys = MLX::Core.concatenate([key_cache, keys], 2)
        values = MLX::Core.concatenate([value_cache, values], 2)
      end

      scores = MLX::Core.matmul(queries, MLX::Core.transpose(keys, [0, 1, 3, 2]))
      scores = MLX::Core.add(scores, mask.astype(scores.dtype)) unless mask.nil?
      scores = MLX::Core.softmax(scores.astype(MLX::Core.float32), -1).astype(scores.dtype)
      values_hat = MLX::Core.matmul(scores, values)
      values_hat = MLX::Core.transpose(values_hat, [0, 2, 1, 3])
      values_hat = MLX::Core.reshape(values_hat, [batch_size, q_len, @num_heads * @head_dim])
      [out_proj.call(values_hat), [keys, values]]
    end
  end

  class DenseActivation < MLX::NN::Module
    def initialize(config)
      super()
      mlp_dims = config.d_ff || (config.d_model * 4)
      @gated = config.feed_forward_proj.start_with?("gated")
      if @gated
        self.wi_0 = MLX::NN::Linear.new(config.d_model, mlp_dims, bias: false)
        self.wi_1 = MLX::NN::Linear.new(config.d_model, mlp_dims, bias: false)
      else
        self.wi = MLX::NN::Linear.new(config.d_model, mlp_dims, bias: false)
      end
      self.wo = MLX::NN::Linear.new(mlp_dims, config.d_model, bias: false)

      activation = config.feed_forward_proj.sub(/\Agated-/, "")
      @act = case activation
      when "relu"
        ->(x) { MLX::NN.relu(x) }
      when "gelu"
        ->(x) { MLX::NN.gelu(x) }
      when "silu"
        ->(x) { MLX::NN.silu(x) }
      else
        raise ArgumentError, "Unknown activation: #{activation}"
      end
    end

    def call(x)
      if @gated
        hidden_act = @act.call(wi_0.call(x))
        hidden_linear = wi_1.call(x)
        x = MLX::Core.multiply(hidden_act, hidden_linear)
      else
        x = @act.call(wi.call(x))
      end
      wo.call(x)
    end
  end

  class TransformerEncoderLayer < MLX::NN::Module
    def initialize(config)
      super()
      self.attention = MultiHeadAttention.new(config)
      self.ln1 = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.ln2 = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.dense = DenseActivation.new(config)
    end

    def call(x, mask)
      y = ln1.call(x)
      y, = attention.call(y, y, y, mask: mask)
      x = MLX::Core.add(x, y)

      y = ln2.call(x)
      y = dense.call(y)
      MLX::Core.add(x, y)
    end
  end

  class TransformerEncoder < MLX::NN::Module
    def initialize(config)
      super()
      self.layers = Array.new(config.num_layers) { TransformerEncoderLayer.new(config) }
      self.ln = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.relative_attention_bias = RelativePositionBias.new(config, true)
    end

    def call(x)
      pos_bias = relative_attention_bias.call(x.shape[1], x.shape[1])
      layers.each do |layer|
        x = layer.call(x, pos_bias)
      end
      ln.call(x)
    end
  end

  class TransformerDecoderLayer < MLX::NN::Module
    def initialize(config)
      super()
      self.self_attention = MultiHeadAttention.new(config)
      self.cross_attention = MultiHeadAttention.new(config)
      self.ln1 = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.ln2 = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.ln3 = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.dense = DenseActivation.new(config)
    end

    def call(x, memory, mask, memory_mask, cache: nil)
      y = ln1.call(x)
      y, new_cache = self_attention.call(y, y, y, mask: mask, cache: cache)
      x = MLX::Core.add(x, y)

      y = ln2.call(x)
      y, = cross_attention.call(y, memory, memory, mask: memory_mask)
      x = MLX::Core.add(x, y)

      y = ln3.call(x)
      y = dense.call(y)
      x = MLX::Core.add(x, y)
      [x, new_cache]
    end
  end

  class TransformerDecoder < MLX::NN::Module
    def initialize(config)
      super()
      self.layers = Array.new(config.num_decoder_layers) { TransformerDecoderLayer.new(config) }
      self.ln = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.relative_attention_bias = RelativePositionBias.new(config, false)
    end

    def call(x, memory, cache: nil)
      cache ||= Array.new(layers.length)
      offset = if !cache[0].nil?
        cache[0][0].shape[2]
      else
        0
      end

      t = x.shape[1]
      mask = if t > 1
        T5Example.create_additive_causal_mask(t, offset: offset)
      else
        nil
      end

      pos_bias = relative_attention_bias.call(t + offset, t + offset, offset: offset)
      mask = if mask.nil?
        pos_bias
      else
        MLX::Core.add(mask, pos_bias)
      end

      layers.each_with_index do |layer, e|
        x, cache[e] = layer.call(x, memory, mask, nil, cache: cache[e])
      end
      [ln.call(x), cache]
    end
  end

  class OutputHead < MLX::NN::Module
    def initialize(config)
      super()
      self.linear = MLX::NN::Linear.new(config.d_model, config.vocab_size, bias: false)
    end

    def call(inputs)
      linear.call(inputs)
    end
  end

  class Model < MLX::NN::Module
    attr_reader :tie_word_embeddings
    attr_accessor :cache

    def initialize(config)
      super()
      self.wte = MLX::NN::Embedding.new(config.vocab_size, config.d_model)
      self.encoder = TransformerEncoder.new(config)
      self.decoder = TransformerDecoder.new(config)
      @tie_word_embeddings = config.tie_word_embeddings
      self.lm_head = OutputHead.new(config) unless @tie_word_embeddings
      @model_dim = config.d_model
      reset_cache
    end

    def encode(inputs)
      encoder.call(wte.call(inputs))
    end

    def decode(inputs, memory, cache: nil)
      inputs = wte.call(inputs)
      y, next_cache = decoder.call(inputs, memory, cache: cache)
      logits = if !tie_word_embeddings
        lm_head.call(y)
      else
        y = MLX::Core.multiply(y, @model_dim**-0.5)
        MLX::Core.matmul(y, wte.weight.T)
      end
      [logits, next_cache]
    end

    def decode_stateful(inputs, memory)
      logits, self.cache = decode(inputs, memory, cache: cache)
      logits
    end

    def truncate_cache(num_to_truncate)
      return if num_to_truncate <= 0
      return if cache[0].nil?

      cache_length = cache[0][0].shape[2]
      if num_to_truncate >= cache_length
        reset_cache
        return
      end

      keep = cache_length - num_to_truncate
      indices = MLX::Core.array((0...keep).to_a, MLX::Core.int32)
      self.cache = cache.map do |layer_cache|
        next nil if layer_cache.nil?

        key_cache, value_cache = layer_cache
        [
          MLX::Core.take(key_cache, indices, 2),
          MLX::Core.take(value_cache, indices, 2)
        ]
      end
    end

    def reset_cache
      self.cache = Array.new(decoder.layers.length)
    end

    def call(inputs, decoder_inputs)
      memory = encode(inputs)
      logits, = decode(decoder_inputs, memory, cache: nil)
      logits
    end
  end

  def generate(prompt_ids, model, decoder_start_id:, temp: 0.0)
    sample = lambda do |logits|
      if temp.to_f.zero?
        MLX::Core.argmax(logits, -1)
      else
        MLX::Core.categorical(MLX::Core.multiply(logits, 1.0 / temp.to_f))
      end
    end

    prompt = prompt_ids
    prompt = MLX::Core.expand_dims(prompt, 0) if prompt.ndim == 1
    memory = model.encode(prompt)
    cache = nil
    y = MLX::Core.array([decoder_start_id], MLX::Core.int32)

    Enumerator.new do |enum|
      loop do
        logits, cache = model.decode(MLX::Core.expand_dims(y, 0), memory, cache: cache)
        last_idx = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
        last = MLX::Core.take(logits, last_idx, 1)
        last = MLX::Core.squeeze(last, 1)
        y = sample.call(last)
        enum << MLX::Core.squeeze(y)
      end
    end
  end
end
