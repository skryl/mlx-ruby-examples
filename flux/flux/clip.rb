# frozen_string_literal: true


require "mlx"
require "mlx/dsl"

module FluxExample
  class CLIPTextModelConfig
    include MLX::DSL::ConfigSchema

    field :num_layers, Integer, default: 6
    field :model_dims, Integer, default: 768
    field :num_heads, Integer, default: 12
    field :max_length, Integer, default: 77
    field :vocab_size, Integer, default: 49_408
    field :hidden_act, String, default: "quick_gelu"

    def self.from_dict(config)
      p = config.transform_keys(&:to_s)
      from_hash(
        num_layers: p.fetch("num_hidden_layers", p.fetch("num_layers", 6)),
        model_dims: p.fetch("hidden_size", p.fetch("model_dims", 768)),
        num_heads: p.fetch("num_attention_heads", p.fetch("num_heads", 12)),
        max_length: p.fetch("max_position_embeddings", p.fetch("max_length", 77)),
        vocab_size: p.fetch("vocab_size", 49_408),
        hidden_act: p.fetch("hidden_act", "quick_gelu")
      )
    end
  end

  class CLIPOutput
    attr_reader :pooled_output, :last_hidden_state, :hidden_states

    def initialize(pooled_output:, last_hidden_state:, hidden_states:)
      @pooled_output = pooled_output
      @last_hidden_state = last_hidden_state
      @hidden_states = hidden_states
    end
  end

  class CLIPEncoderLayer < MLX::NN::Module
    def initialize(model_dims, num_heads, activation)
      super()
      @activation = activation

      self.layer_norm1 = MLX::NN::LayerNorm.new(model_dims)
      self.layer_norm2 = MLX::NN::LayerNorm.new(model_dims)
      self.attention = MLX::NN::MultiHeadAttention.new(model_dims, num_heads, bias: true)
      self.linear1 = MLX::NN::Linear.new(model_dims, 4 * model_dims)
      self.linear2 = MLX::NN::Linear.new(4 * model_dims, model_dims)
    end

    def call(x, attn_mask = nil)
      y = layer_norm1.call(x)
      y = attention.call(y, y, y, attn_mask)
      x = MLX::Core.add(y, x)

      y = layer_norm2.call(x)
      y = linear1.call(y)
      y = activate(y)
      y = linear2.call(y)
      MLX::Core.add(y, x)
    end

    private

    def activate(x)
      if @activation == "gelu"
        MLX::NN.gelu(x)
      else
        MLX::Core.multiply(x, MLX::Core.sigmoid(MLX::Core.multiply(1.702, x)))
      end
    end
  end

  class CLIPTextModel < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config

      self.token_embedding = MLX::NN::Embedding.new(config.vocab_size, config.model_dims)
      self.position_embedding = MLX::NN::Embedding.new(config.max_length, config.model_dims)
      layer_count = [config.num_layers, 8].min
      self.layers = Array.new(layer_count) do
        CLIPEncoderLayer.new(config.model_dims, config.num_heads, config.hidden_act)
      end
      self.final_layer_norm = MLX::NN::LayerNorm.new(config.model_dims)
    end

    def sanitize(weights)
      self.class.weight_mapper.apply(weights)
    end

    def self.weight_mapper
      @weight_mapper ||= MLX::DSL.weight_map do
        strip_prefix "text_model."
        strip_prefix "embeddings."
        strip_prefix "encoder."
        rename "self_attn." => "attention."
        rename "q_proj." => "query_proj."
        rename "k_proj." => "key_proj."
        rename "v_proj." => "value_proj."
        rename "mlp.fc1" => "linear1"
        rename "mlp.fc2" => "linear2"
      end
    end

    def call(x)
      b, n = x.shape
      eos_tokens = MLX::Core.argmax(x, -1)

      h = token_embedding.call(x)
      pos = MLX::Core.slice(position_embedding.weight, [0, 0], [n, position_embedding.weight.shape[1]])
      h = MLX::Core.add(h, pos)

      mask = get_mask(n, h.dtype)
      hidden_states = []
      layers.each do |layer|
        h = layer.call(h, mask)
        hidden_states << h
      end

      h = final_layer_norm.call(h)
      last_hidden_state = h

      pooled = select_eos(h, eos_tokens)
      CLIPOutput.new(
        pooled_output: pooled,
        last_hidden_state: last_hidden_state,
        hidden_states: hidden_states
      )
    end

    private

    def get_mask(n, dtype)
      idx = MLX::Core.arange(0, n, 1)
      lhs = MLX::Core.expand_dims(idx, 1)
      rhs = MLX::Core.expand_dims(idx, 0)
      mask = MLX::Core.less(lhs, rhs).astype(dtype)
      scale = (dtype == MLX::Core.float16 ? -6e4 : -1e9)
      MLX::Core.multiply(mask, scale)
    end

    def select_eos(x, eos_tokens)
      b = x.shape[0]
      d = x.shape[2]
      idx = eos_tokens.astype(MLX::Core.int32)
      idx = MLX::Core.maximum(idx, 0)
      idx = MLX::Core.minimum(idx, x.shape[1] - 1)
      idx = MLX::Core.reshape(idx, [b, 1, 1])
      idx = MLX::Core.broadcast_to(idx, [b, 1, d])
      selected = MLX::Core.take_along_axis(x, idx, 1)
      MLX::Core.squeeze(selected, 1)
    end
  end
end
