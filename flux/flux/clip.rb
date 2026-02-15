# frozen_string_literal: true


require "mlx"

module FluxExample
  class CLIPTextModelConfig
    attr_reader :num_layers, :model_dims, :num_heads, :max_length, :vocab_size, :hidden_act

    def initialize(
      num_layers: 6,
      model_dims: 768,
      num_heads: 12,
      max_length: 77,
      vocab_size: 49_408,
      hidden_act: "quick_gelu"
    )
      @num_layers = num_layers
      @model_dims = model_dims
      @num_heads = num_heads
      @max_length = max_length
      @vocab_size = vocab_size
      @hidden_act = hidden_act
    end

    def self.from_dict(config)
      p = config.transform_keys(&:to_s)
      new(
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
      weights.each_with_object({}) do |(key, value), out|
        key = key.to_s
        key = key.delete_prefix("text_model.")
        key = key.delete_prefix("embeddings.")
        key = key.delete_prefix("encoder.")
        key = key.gsub("self_attn.", "attention.")
        key = key.gsub("q_proj.", "query_proj.")
        key = key.gsub("k_proj.", "key_proj.")
        key = key.gsub("v_proj.", "value_proj.")
        key = key.gsub("mlp.fc1", "linear1")
        key = key.gsub("mlp.fc2", "linear2")
        out[key] = value
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
      out = []
      x_arr = x.to_a
      eos_arr = eos_tokens.to_a
      b.times do |i|
        idx = eos_arr[i].to_i
        idx = [[idx, 0].max, x.shape[1] - 1].min
        out << x_arr[i][idx]
      end
      MLX::Core.array(out, x.dtype).reshape([b, d])
    end
  end
end
