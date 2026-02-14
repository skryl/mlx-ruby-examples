# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module LlavaExample
  class VisionConfig
    attr_reader :model_type,
                :num_hidden_layers,
                :hidden_size,
                :intermediate_size,
                :num_attention_heads,
                :image_size,
                :patch_size,
                :projection_dim,
                :vocab_size,
                :num_channels,
                :layer_norm_eps

    def initialize(
      model_type: "clip_vision_model",
      num_hidden_layers: 24,
      hidden_size: 1024,
      intermediate_size: 4096,
      num_attention_heads: 16,
      image_size: 336,
      patch_size: 14,
      projection_dim: 768,
      vocab_size: 32_000,
      num_channels: 3,
      layer_norm_eps: 1e-5
    )
      @model_type = model_type
      @num_hidden_layers = num_hidden_layers
      @hidden_size = hidden_size
      @intermediate_size = intermediate_size
      @num_attention_heads = num_attention_heads
      @image_size = image_size
      @patch_size = patch_size
      @projection_dim = projection_dim
      @vocab_size = vocab_size
      @num_channels = num_channels
      @layer_norm_eps = layer_norm_eps
    end

    def self.from_dict(params)
      p = params.transform_keys(&:to_s)
      new(
        model_type: p.fetch("model_type", "clip_vision_model"),
        num_hidden_layers: p.fetch("num_hidden_layers", 24),
        hidden_size: p.fetch("hidden_size", 1024),
        intermediate_size: p.fetch("intermediate_size", 4096),
        num_attention_heads: p.fetch("num_attention_heads", 16),
        image_size: p.fetch("image_size", 336),
        patch_size: p.fetch("patch_size", 14),
        projection_dim: p.fetch("projection_dim", 768),
        vocab_size: p.fetch("vocab_size", 32_000),
        num_channels: p.fetch("num_channels", 3),
        layer_norm_eps: p.fetch("layer_norm_eps", 1e-5)
      )
    end
  end

  class VisionAttention < MLX::NN::Module
    def initialize(
      dims:,
      num_heads:,
      query_input_dims: nil,
      key_input_dims: nil,
      value_input_dims: nil,
      value_dims: nil,
      value_output_dims: nil,
      bias: false
    )
      super()

      if (dims % num_heads) != 0
        raise ArgumentError, "input dims must be divisible by number of heads (#{dims} % #{num_heads} != 0)"
      end

      query_input_dims ||= dims
      key_input_dims ||= dims
      value_input_dims ||= key_input_dims
      value_dims ||= dims
      value_output_dims ||= dims

      @num_heads = num_heads

      self.q_proj = MLX::NN::Linear.new(query_input_dims, dims, bias: bias)
      self.k_proj = MLX::NN::Linear.new(key_input_dims, dims, bias: bias)
      self.v_proj = MLX::NN::Linear.new(value_input_dims, value_dims, bias: bias)
      self.out_proj = MLX::NN::Linear.new(value_dims, value_output_dims, bias: bias)
    end

    def call(queries, keys, values, mask: nil)
      queries = q_proj.call(queries)
      keys = k_proj.call(keys)
      values = v_proj.call(values)

      batch_size, q_len, _q_dim = queries.shape
      _batch_size2, k_len, _k_dim = keys.shape
      q_head_dim = queries.shape[2] / @num_heads
      k_head_dim = keys.shape[2] / @num_heads
      v_head_dim = values.shape[2] / @num_heads

      queries = MLX::Core.transpose(
        MLX::Core.reshape(queries, [batch_size, q_len, @num_heads, q_head_dim]),
        [0, 2, 1, 3]
      )
      keys = MLX::Core.transpose(
        MLX::Core.reshape(keys, [batch_size, k_len, @num_heads, k_head_dim]),
        [0, 2, 3, 1]
      )
      values = MLX::Core.transpose(
        MLX::Core.reshape(values, [batch_size, k_len, @num_heads, v_head_dim]),
        [0, 2, 1, 3]
      )

      scale = Math.sqrt(1.0 / queries.shape[-1])
      scores = MLX::Core.matmul(MLX::Core.multiply(queries, scale), keys)
      scores = MLX::Core.add(scores, mask.astype(scores.dtype)) unless mask.nil?
      scores = MLX::Core.softmax(scores, -1)
      attended = MLX::Core.matmul(scores, values)
      attended = MLX::Core.transpose(attended, [0, 2, 1, 3])
      attended = MLX::Core.reshape(attended, [batch_size, q_len, @num_heads * v_head_dim])
      out_proj.call(attended)
    end
  end

  class VisionMLP < MLX::NN::Module
    def initialize(config)
      super()
      self.gelu = MLX::NN::GELU.new
      self.fc1 = MLX::NN::Linear.new(config.hidden_size, config.intermediate_size)
      self.fc2 = MLX::NN::Linear.new(config.intermediate_size, config.hidden_size)
    end

    def call(x)
      fc2.call(gelu.call(fc1.call(x)))
    end
  end

  class EncoderLayer < MLX::NN::Module
    def initialize(config)
      super()
      @embed_dim = config.hidden_size
      self.self_attn = VisionAttention.new(
        dims: config.hidden_size,
        num_heads: config.num_attention_heads,
        bias: true
      )
      self.layer_norm1 = MLX::NN::LayerNorm.new(@embed_dim, eps: config.layer_norm_eps)
      self.mlp = VisionMLP.new(config)
      self.layer_norm2 = MLX::NN::LayerNorm.new(@embed_dim, eps: config.layer_norm_eps)
    end

    def call(x, mask: nil)
      y = layer_norm1.call(x)
      y = self_attn.call(y, y, y, mask: mask)
      x = MLX::Core.add(x, y)
      y = layer_norm2.call(x)
      y = mlp.call(y)
      MLX::Core.add(x, y)
    end
  end

  class Encoder < MLX::NN::Module
    def initialize(config)
      super()
      self.layers = Array.new(config.num_hidden_layers) { EncoderLayer.new(config) }
    end
  end

  class VisionEmbeddings < MLX::NN::Module
    def initialize(config)
      super()
      @config = config
      @embed_dim = config.hidden_size
      @image_size = config.image_size
      @patch_size = config.patch_size

      self.class_embedding = MLX::Core.zeros([config.hidden_size])
      self.patch_embedding = MLX::NN::Conv2d.new(
        config.num_channels,
        @embed_dim,
        @patch_size,
        stride: @patch_size,
        bias: false
      )

      @num_patches = (@image_size / @patch_size)**2
      @num_positions = @num_patches + 1
      self.position_embedding = MLX::NN::Embedding.new(@num_positions, @embed_dim)
    end

    def call(x)
      batch_size = x.shape[0]
      patch_embeddings = patch_embedding.call(x)
      h = patch_embeddings.shape[1]
      w = patch_embeddings.shape[2]
      patch_embeddings = MLX::Core.reshape(patch_embeddings, [batch_size, h * w, @embed_dim])

      cls_embeddings = MLX::Core.broadcast_to(class_embedding, [batch_size, 1, @embed_dim])
      embeddings = MLX::Core.concatenate([cls_embeddings, patch_embeddings], 1)
      pos = MLX::Core.expand_dims(position_embedding.weight, 0)
      MLX::Core.add(embeddings, pos)
    end
  end

  class ClipVisionModel < MLX::NN::Module
    def initialize(config)
      super()
      self.embeddings = VisionEmbeddings.new(config)
      # Keep this field name to match HF LLaVA checkpoint keys.
      self.pre_layrnorm = MLX::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
      self.encoder = Encoder.new(config)
      self.post_layernorm = MLX::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
    end

    def call(x, output_hidden_states: false)
      x = embeddings.call(x)
      x = pre_layrnorm.call(x)

      hidden_states = output_hidden_states ? [x] : nil

      encoder.layers.each do |layer|
        x = layer.call(x, mask: nil)
        hidden_states << x if output_hidden_states
      end

      pooler_output = post_layernorm.call(
        MLX::Core.squeeze(
          MLX::Core.take(x, MLX::Core.array([0], MLX::Core.int32), 1),
          1
        )
      )
      [pooler_output, x, hidden_states]
    end
  end

  class VisionModel < MLX::NN::Module
    def initialize(config)
      super()

      @model_type = config.model_type
      raise ArgumentError, "Unsupported vision model type: #{@model_type}" unless @model_type == "clip_vision_model"

      self.vision_model = ClipVisionModel.new(config)
    end

    def call(x, output_hidden_states: false)
      vision_model.call(x, output_hidden_states: output_hidden_states)
    end

    def self.sanitize(weights)
      weights.each_with_object({}) do |(key, value), out|
        key = key.to_s
        if key.include?("position_ids")
          next
        elsif key.include?("patch_embedding.weight") && value.shape.length == 4
          out[key] = MLX::Core.transpose(value, [0, 2, 3, 1])
        else
          out[key] = value
        end
      end
    end
  end
end
