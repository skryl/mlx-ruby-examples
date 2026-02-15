# frozen_string_literal: true


require "mlx"

module LlavaExample
  class TextConfig
    attr_accessor :model_type,
                  :hidden_size,
                  :num_hidden_layers,
                  :intermediate_size,
                  :num_attention_heads,
                  :rms_norm_eps,
                  :vocab_size,
                  :num_key_value_heads,
                  :rope_theta,
                  :rope_traditional,
                  :rope_scaling

    def initialize(
      model_type: "llama",
      hidden_size: 4096,
      num_hidden_layers: 32,
      intermediate_size: 11_008,
      num_attention_heads: 32,
      rms_norm_eps: 1e-6,
      vocab_size: 32_000,
      num_key_value_heads: nil,
      rope_theta: 10_000.0,
      rope_traditional: false,
      rope_scaling: nil
    )
      @model_type = model_type
      @hidden_size = hidden_size
      @num_hidden_layers = num_hidden_layers
      @intermediate_size = intermediate_size
      @num_attention_heads = num_attention_heads
      @rms_norm_eps = rms_norm_eps
      @vocab_size = vocab_size
      @num_key_value_heads = num_key_value_heads || num_attention_heads
      @rope_theta = rope_theta
      @rope_traditional = rope_traditional
      @rope_scaling = rope_scaling
      validate_rope_scaling!
    end

    def self.from_dict(params)
      p = params.transform_keys(&:to_s)
      new(
        model_type: p.fetch("model_type", "llama"),
        hidden_size: p.fetch("hidden_size", 4096),
        num_hidden_layers: p.fetch("num_hidden_layers", 32),
        intermediate_size: p.fetch("intermediate_size", 11_008),
        num_attention_heads: p.fetch("num_attention_heads", 32),
        rms_norm_eps: p.fetch("rms_norm_eps", 1e-6),
        vocab_size: p.fetch("vocab_size", 32_000),
        num_key_value_heads: p["num_key_value_heads"],
        rope_theta: p.fetch("rope_theta", 10_000.0),
        rope_traditional: p.fetch("rope_traditional", false),
        rope_scaling: p["rope_scaling"]
      )
    end

    def to_h
      {
        "model_type" => model_type,
        "hidden_size" => hidden_size,
        "num_hidden_layers" => num_hidden_layers,
        "intermediate_size" => intermediate_size,
        "num_attention_heads" => num_attention_heads,
        "rms_norm_eps" => rms_norm_eps,
        "vocab_size" => vocab_size,
        "num_key_value_heads" => num_key_value_heads,
        "rope_theta" => rope_theta,
        "rope_traditional" => rope_traditional,
        "rope_scaling" => rope_scaling
      }
    end

    private

    def validate_rope_scaling!
      return if rope_scaling.nil?

      required = %w[factor type]
      unless required.all? { |key| rope_scaling.key?(key) || rope_scaling.key?(key.to_sym) }
        raise ArgumentError, "rope_scaling must contain keys #{required.inspect}"
      end
      type = rope_scaling["type"] || rope_scaling[:type]
      return if type == "linear"

      raise ArgumentError, "rope_scaling 'type' currently only supports 'linear'"
    end
  end

  class Attention < MLX::NN::Module
    def initialize(config)
      super()
      dim = config.hidden_size
      @n_heads = config.num_attention_heads
      @n_kv_heads = config.num_key_value_heads
      @head_dim = config.hidden_size / @n_heads
      @repeats = @n_heads / @n_kv_heads
      @scale = @head_dim**-0.5

      self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
      self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
      self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
      self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

      rope_scale = if !config.rope_scaling.nil? && (config.rope_scaling["type"] || config.rope_scaling[:type]) == "linear"
        1.0 / (config.rope_scaling["factor"] || config.rope_scaling[:factor]).to_f
      else
        1.0
      end
      self.rope = MLX::NN::RoPE.new(
        @head_dim,
        traditional: config.rope_traditional,
        base: config.rope_theta,
        scale: rope_scale
      )
    end

    def call(x, mask: nil, cache: nil)
      bsz, seq_len, _dims = x.shape

      queries = q_proj.call(x)
      keys = k_proj.call(x)
      values = v_proj.call(x)

      queries = MLX::Core.transpose(MLX::Core.reshape(queries, [bsz, seq_len, @n_heads, @head_dim]), [0, 2, 1, 3])
      keys = MLX::Core.transpose(MLX::Core.reshape(keys, [bsz, seq_len, @n_kv_heads, @head_dim]), [0, 2, 1, 3])
      values = MLX::Core.transpose(MLX::Core.reshape(values, [bsz, seq_len, @n_kv_heads, @head_dim]), [0, 2, 1, 3])

      if !cache.nil?
        key_cache, value_cache = cache
        queries = rope.call(queries, offset: key_cache.shape[2])
        keys = rope.call(keys, offset: key_cache.shape[2])
        keys = MLX::Core.concatenate([key_cache, keys], 2)
        values = MLX::Core.concatenate([value_cache, values], 2)
      else
        queries = rope.call(queries)
        keys = rope.call(keys)
      end

      if @n_kv_heads != @n_heads
        keys = repeat_heads(keys, @repeats)
        values = repeat_heads(values, @repeats)
      end

      out = MLX::Core.scaled_dot_product_attention(queries, keys, values, @scale, mask)
      out = MLX::Core.transpose(out, [0, 2, 1, 3])
      out = MLX::Core.reshape(out, [bsz, seq_len, @n_heads * @head_dim])
      [o_proj.call(out), [keys, values]]
    end

    private

    def repeat_heads(x, repeats)
      b, h, t, d = x.shape
      expanded = MLX::Core.expand_dims(x, 2)
      repeated = MLX::Core.concatenate(Array.new(repeats, expanded), 2)
      MLX::Core.reshape(repeated, [b, h * repeats, t, d])
    end
  end

  class MLP < MLX::NN::Module
    def initialize(dim, hidden_dim)
      super()
      self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
      self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
      self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
    end

    def call(x)
      down_proj.call(MLX::Core.multiply(MLX::NN.silu(gate_proj.call(x)), up_proj.call(x)))
    end
  end

  class TransformerBlock < MLX::NN::Module
    def initialize(config)
      super()
      self.self_attn = Attention.new(config)
      self.mlp = MLP.new(config.hidden_size, config.intermediate_size)
      self.input_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
      self.post_attention_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
    end

    def call(x, mask: nil, cache: nil)
      residual, cache = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
      h = MLX::Core.add(x, residual)
      residual = mlp.call(post_attention_layernorm.call(h))
      [MLX::Core.add(h, residual), cache]
    end
  end

  class Llama < MLX::NN::Module
    def initialize(config)
      super()
      @config = config
      self.embed_tokens = MLX::NN::Embedding.new(config.vocab_size, config.hidden_size)
      self.layers = Array.new(config.num_hidden_layers) { TransformerBlock.new(config) }
      self.norm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
    end

    def call(inputs, cache: nil, inputs_embeds: nil)
      h = inputs_embeds.nil? ? embed_tokens.call(inputs) : inputs_embeds

      mask = nil
      if h.shape[1] > 1
        mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)
      end

      cache ||= Array.new(layers.length)
      layers.each_with_index do |layer, i|
        h, cache[i] = layer.call(h, mask: mask, cache: cache[i])
      end
      [norm.call(h), cache]
    end
  end

  class LanguageModel < MLX::NN::Module
    attr_reader :model_type

    def initialize(config)
      super()
      @model_type = config.model_type
      raise ArgumentError, "Only llama text backend is supported" unless @model_type == "llama"

      self.model = Llama.new(config)
      self.lm_head = MLX::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)
    end

    def call(inputs, cache: nil, inputs_embeds: nil)
      out, cache = model.call(inputs, cache: cache, inputs_embeds: inputs_embeds)
      [lm_head.call(out), cache]
    end

    def self.sanitize(weights)
      weights.each_with_object({}) do |(key, value), out|
        next if key.to_s.include?("self_attn.rotary_emb.inv_freq")

        out[key.to_s] = value
      end
    end
  end
end
