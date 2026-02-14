# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("../..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module FluxExample
  class T5Config
    attr_reader :vocab_size,
                :num_layers,
                :num_heads,
                :relative_attention_num_buckets,
                :d_kv,
                :d_model,
                :feed_forward_proj,
                :tie_word_embeddings,
                :d_ff,
                :num_decoder_layers,
                :relative_attention_max_distance,
                :layer_norm_epsilon

    def initialize(
      vocab_size: 32_128,
      num_layers: 8,
      num_heads: 8,
      relative_attention_num_buckets: 32,
      d_kv: 64,
      d_model: 1024,
      feed_forward_proj: "gelu",
      tie_word_embeddings: false,
      d_ff: nil,
      num_decoder_layers: nil,
      relative_attention_max_distance: 128,
      layer_norm_epsilon: 1e-6
    )
      @vocab_size = vocab_size
      @num_layers = num_layers
      @num_heads = num_heads
      @relative_attention_num_buckets = relative_attention_num_buckets
      @d_kv = d_kv
      @d_model = d_model
      @feed_forward_proj = feed_forward_proj
      @tie_word_embeddings = tie_word_embeddings
      @d_ff = d_ff || (4 * d_model)
      @num_decoder_layers = num_decoder_layers || num_layers
      @relative_attention_max_distance = relative_attention_max_distance
      @layer_norm_epsilon = layer_norm_epsilon
    end

    def self.from_dict(config)
      p = config.transform_keys(&:to_s)
      new(
        vocab_size: p.fetch("vocab_size"),
        num_layers: p.fetch("num_layers"),
        num_heads: p.fetch("num_heads"),
        relative_attention_num_buckets: p.fetch("relative_attention_num_buckets"),
        d_kv: p.fetch("d_kv"),
        d_model: p.fetch("d_model"),
        feed_forward_proj: p.fetch("feed_forward_proj"),
        tie_word_embeddings: p.fetch("tie_word_embeddings"),
        d_ff: p["d_ff"],
        num_decoder_layers: p["num_decoder_layers"],
        relative_attention_max_distance: p.fetch("relative_attention_max_distance", 128),
        layer_norm_epsilon: p.fetch("layer_norm_epsilon", 1e-6)
      )
    end
  end

  class T5EncoderLayer < MLX::NN::Module
    def initialize(config)
      super()
      self.ln1 = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.ln2 = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
      self.attn = MLX::NN::MultiHeadAttention.new(config.d_model, config.num_heads, bias: false)
      self.ff1 = MLX::NN::Linear.new(config.d_model, config.d_ff, bias: false)
      self.ff2 = MLX::NN::Linear.new(config.d_ff, config.d_model, bias: false)
    end

    def call(x, mask)
      y = ln1.call(x)
      y = attn.call(y, y, y, mask)
      x = MLX::Core.add(x, y)

      y = ln2.call(x)
      y = MLX::NN.gelu(ff1.call(y))
      y = ff2.call(y)
      MLX::Core.add(x, y)
    end
  end

  class T5Encoder < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config
      self.wte = MLX::NN::Embedding.new(config.vocab_size, config.d_model)
      layer_count = [config.num_layers, 8].min
      self.layers = Array.new(layer_count) { T5EncoderLayer.new(config) }
      self.ln = MLX::NN::RMSNorm.new(config.d_model, eps: config.layer_norm_epsilon)
    end

    def sanitize(weights)
      replacements = [
        [".block.", ".layers."],
        [".k.", ".key_proj."],
        [".o.", ".out_proj."],
        [".q.", ".query_proj."],
        [".v.", ".value_proj."],
        ["shared.", "wte."],
        ["lm_head.", "lm_head.linear."],
        [".layer.0.layer_norm.", ".ln1."],
        [".layer.1.layer_norm.", ".ln2."],
        [".layer.2.layer_norm.", ".ln3."],
        [".final_layer_norm.", ".ln."],
        [".layer.0.SelfAttention.", ".attention."],
        [".layer.1.DenseReluDense.", ".dense."]
      ]

      weights.each_with_object({}) do |(key, value), out|
        k = key.to_s
        replacements.each { |old, newv| k = k.gsub(old, newv) }
        out[k] = value
      end
    end

    def call(inputs)
      h = wte.call(inputs)
      mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(h.shape[1])
      mask = mask.astype(h.dtype)
      layers.each { |layer| h = layer.call(h, mask) }
      ln.call(h)
    end
  end
end
