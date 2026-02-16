# frozen_string_literal: true


require "mlx"
require "mlx/dsl"

module FluxExample
  class T5Config
    include MLX::DSL::ConfigSchema

    field :vocab_size, Integer, default: 32_128
    field :num_layers, Integer, default: 8
    field :num_heads, Integer, default: 8
    field :relative_attention_num_buckets, Integer, default: 32
    field :d_kv, Integer, default: 64
    field :d_model, Integer, default: 1024
    field :feed_forward_proj, String, default: "gelu"
    field :tie_word_embeddings, [TrueClass, FalseClass], default: false
    field :d_ff, Integer, default: ->(cfg) { cfg.d_model * 4 }
    field :num_decoder_layers, Integer, default: ->(cfg) { cfg.num_layers }
    field :relative_attention_max_distance, Integer, default: 128
    field :layer_norm_epsilon, [Integer, Float], default: 1e-6

    def self.from_dict(config)
      p = config.transform_keys(&:to_s)
      from_hash(
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

    def call(x, mask = nil, **kwargs)
      mask = kwargs[:mask] if kwargs.key?(:mask)
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
      self.class.weight_mapper.apply(weights)
    end

    def self.weight_mapper
      @weight_mapper ||= MLX::DSL.weight_map do
        rename ".block." => ".layers."
        rename ".k." => ".key_proj."
        rename ".o." => ".out_proj."
        rename ".q." => ".query_proj."
        rename ".v." => ".value_proj."
        rename "shared." => "wte."
        rename "lm_head." => "lm_head.linear."
        rename ".layer.0.layer_norm." => ".ln1."
        rename ".layer.1.layer_norm." => ".ln2."
        rename ".layer.2.layer_norm." => ".ln3."
        rename ".final_layer_norm." => ".ln."
        rename ".layer.0.SelfAttention." => ".attention."
        rename ".layer.1.DenseReluDense." => ".dense."
      end
    end

    def call(inputs)
      h = wte.call(inputs)
      mask = MLX::DSL::Masks.causal(length: h.shape[1], dtype: h.dtype)
      h = MLX::DSL.run_stack(layers, h, mask: mask)
      ln.call(h)
    end
  end
end
