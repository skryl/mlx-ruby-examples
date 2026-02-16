# frozen_string_literal: true

require "json"

ROOT = File.expand_path("..", __dir__)

require "mlx"
require "mlx/dsl"

module LoraExample
  class ModelArgs
    include MLX::DSL::ConfigSchema

    field :hidden_size, Integer, required: true
    field :num_hidden_layers, Integer, required: true
    field :intermediate_size, Integer, required: true
    field :num_attention_heads, Integer, required: true
    field :rms_norm_eps, [Integer, Float], required: true
    field :vocab_size, Integer, required: true
    field :num_key_value_heads, Integer, default: ->(cfg) { cfg.num_attention_heads }
    field :rope_theta, [Integer, Float], default: 10_000.0
    field :rope_traditional, [TrueClass, FalseClass], default: false
    field :model_type, [String, NilClass], default: nil
    field :rope_scaling, [Hash, NilClass], default: nil do |value|
      next if value.nil?

      required_keys = %w[factor type]
      unless required_keys.all? { |key| value.key?(key) || value.key?(key.to_sym) }
        raise ArgumentError, "rope_scaling must contain keys #{required_keys.inspect}"
      end
      scaling_type = value["type"] || value[:type]
      raise ArgumentError, "rope_scaling type only supports linear" unless scaling_type == "linear"
    end

    def self.from_dict(params)
      from_hash(params)
    end
  end

  class LoRALinear < MLX::NN::Module
    def self.from_linear(linear, rank: 8)
      output_dims, input_dims = linear.weight.shape
      if linear.is_a?(MLX::NN::QuantizedLinear)
        input_dims *= (32 / linear.bits)
      end
      lora_linear = new(input_dims, output_dims, lora_rank: rank)
      lora_linear.linear = linear
      lora_linear
    end

    def to_linear
      linear_layer = linear
      bias_exists = linear_layer.respond_to?(:bias) && !linear_layer.bias.nil?
      weight = linear_layer.weight
      is_quantized = linear_layer.is_a?(MLX::NN::QuantizedLinear)

      dtype = weight.dtype
      if is_quantized
        dtype = MLX::Core.float16
        weight = MLX::Core.dequantize(
          weight,
          linear_layer.scales,
          linear_layer.biases,
          linear_layer.group_size,
          linear_layer.bits
        )
      end

      output_dims, input_dims = weight.shape
      fused = MLX::NN::Linear.new(input_dims, output_dims, bias: bias_exists)

      delta = MLX::Core.matmul(lora_a, lora_b)
      delta = MLX::Core.transpose(delta, [1, 0]).astype(dtype)
      delta = MLX::Core.multiply(scale, delta)
      fused.weight = MLX::Core.add(weight, delta)
      fused.bias = linear_layer.bias if bias_exists

      if is_quantized
        fused = MLX::NN::QuantizedLinear.from_linear(
          fused,
          linear_layer.group_size,
          linear_layer.bits
        )
      end
      fused
    end

    def initialize(
      input_dims,
      output_dims,
      lora_rank: 8,
      bias: false,
      scale: 20.0
    )
      super()
      self.linear = MLX::NN::Linear.new(input_dims, output_dims, bias: bias)
      self.scale = scale

      init_scale = 1.0 / Math.sqrt(input_dims)
      self.lora_a = MLX::Core.random_uniform(
        [input_dims, lora_rank],
        -init_scale,
        init_scale,
        MLX::Core.float32
      )
      self.lora_b = MLX::Core.zeros([lora_rank, output_dims])
    end

    def call(x)
      dtype = linear.weight.dtype
      dtype = linear.scales.dtype if linear.is_a?(MLX::NN::QuantizedLinear)
      y = linear.call(x.astype(dtype))
      z = MLX::Core.matmul(MLX::Core.matmul(x, lora_a), lora_b)
      MLX::Core.add(y, MLX::Core.multiply(scale, z))
    end
  end

  class Attention < MLX::NN::Module
    def initialize(args)
      super()

      dim = args.hidden_size
      @n_heads = args.num_attention_heads
      @n_kv_heads = args.num_key_value_heads
      @repeats = @n_heads / @n_kv_heads
      @head_dim = args.hidden_size / @n_heads
      @scale = @head_dim**-0.5

      self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
      self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
      self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
      self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

      rope_factor = if !args.rope_scaling.nil? && (args.rope_scaling["type"] || args.rope_scaling[:type]) == "linear"
        1.0 / (args.rope_scaling["factor"] || args.rope_scaling[:factor]).to_f
      else
        1.0
      end
      self.rope = MLX::NN::RoPE.new(
        @head_dim,
        traditional: args.rope_traditional,
        base: args.rope_theta,
        scale: rope_factor
      )
    end

    def call(x, mask: nil, cache: nil)
      batch_size, seq_len, _dims = x.shape

      queries = q_proj.call(x)
      keys = k_proj.call(x)
      values = v_proj.call(x)

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

      output = MLX::Core.scaled_dot_product_attention(
        queries,
        keys,
        values,
        @scale,
        mask
      )
      output = MLX::Core.transpose(output, [0, 2, 1, 3])
      output = MLX::Core.reshape(output, [batch_size, seq_len, @n_heads * @head_dim])
      [o_proj.call(output), [keys, values]]
    end

    private

    def repeat_heads(array, repeats)
      batch_size, n_heads, seq_len, head_dim = array.shape
      expanded = MLX::Core.expand_dims(array, 2)
      repeated = MLX::Core.concatenate(Array.new(repeats, expanded), 2)
      MLX::Core.reshape(repeated, [batch_size, n_heads * repeats, seq_len, head_dim])
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
    attr_reader :args

    def initialize(args)
      super()
      self.self_attn = Attention.new(args)
      self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
      self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
      self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
      @args = args
    end

    def call(x, mask: nil, cache: nil)
      residual, cache = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
      hidden = MLX::Core.add(x, residual)
      residual = mlp.call(post_attention_layernorm.call(hidden))
      [MLX::Core.add(hidden, residual), cache]
    end
  end

  class LlamaModel < MLX::NN::Module
    attr_reader :args, :vocab_size, :num_hidden_layers

    def initialize(args)
      super()
      @args = args
      @vocab_size = args.vocab_size
      @num_hidden_layers = args.num_hidden_layers
      raise ArgumentError, "vocab_size must be positive" unless @vocab_size.positive?

      self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
      self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
      self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
    end

    def call(inputs, cache: nil)
      hidden = embed_tokens.call(inputs)
      offset = MLX::DSL::Positions.offset_from_cache(cache, layer: 0)

      mask = nil
      if hidden.shape[1] > 1 || offset.positive?
        mask = MLX::DSL::Masks.causal(length: hidden.shape[1], offset: offset, dtype: hidden.dtype)
      end

      cache_state = cache || Array.new(layers.length)
      hidden, next_cache = MLX::DSL.run_stack(layers, hidden, mask: mask, cache: cache_state)

      [norm.call(hidden), next_cache]
    end
  end

  class Model < MLX::NN::Module
    def initialize(args)
      super()
      self.model = LlamaModel.new(args)
      self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
    end

    def call(inputs, cache: nil)
      out, cache = model.call(inputs, cache: cache)
      [lm_head.call(out), cache]
    end
  end
end
