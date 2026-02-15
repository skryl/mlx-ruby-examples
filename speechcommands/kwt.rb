# frozen_string_literal: true


require "mlx"

module SpeechcommandsExample
  class FeedForward < MLX::NN::Module
    def initialize(dim, hidden_dim, dropout: 0.0)
      super()
      self.linear1 = MLX::NN::Linear.new(dim, hidden_dim)
      self.dropout1 = MLX::NN::Dropout.new(dropout)
      self.linear2 = MLX::NN::Linear.new(hidden_dim, dim)
      self.dropout2 = MLX::NN::Dropout.new(dropout)
    end

    def call(x)
      x = linear1.call(x)
      x = MLX::NN.gelu(x)
      x = dropout1.call(x)
      x = linear2.call(x)
      dropout2.call(x)
    end
  end

  class Attention < MLX::NN::Module
    def initialize(dim, heads, dropout: 0.0)
      super()
      @heads = heads
      @scale = dim**-0.5
      self.qkv = MLX::NN::Linear.new(dim, dim * 3, bias: false)
      self.out_linear = MLX::NN::Linear.new(dim, dim)
      self.out_dropout = MLX::NN::Dropout.new(dropout)
    end

    def call(x)
      batch, tokens, dim = x.shape
      head_dim = dim / @heads
      if (head_dim * @heads) != dim
        raise ArgumentError, "dim=#{dim} must be divisible by heads=#{@heads}"
      end

      qkv_tensor = qkv.call(x)
      qkv_tensor = MLX::Core.reshape(qkv_tensor, [batch, tokens, 3, @heads, head_dim])
      qkv_tensor = MLX::Core.transpose(qkv_tensor, [2, 0, 3, 1, 4])
      q = qkv_tensor[0]
      k = qkv_tensor[1]
      v = qkv_tensor[2]

      attn = MLX::Core.matmul(q, MLX::Core.transpose(k, [0, 1, 3, 2]))
      attn = MLX::Core.multiply(attn, @scale)
      attn = MLX::Core.softmax(attn, -1)

      out = MLX::Core.matmul(attn, v)
      out = MLX::Core.transpose(out, [0, 2, 1, 3])
      out = MLX::Core.reshape(out, [batch, tokens, dim])
      out = out_linear.call(out)
      out_dropout.call(out)
    end
  end

  class Block < MLX::NN::Module
    def initialize(dim, heads, mlp_dim, dropout: 0.0)
      super()
      self.attn = Attention.new(dim, heads, dropout: dropout)
      self.norm1 = MLX::NN::LayerNorm.new(dim)
      self.ff = FeedForward.new(dim, mlp_dim, dropout: dropout)
      self.norm2 = MLX::NN::LayerNorm.new(dim)
    end

    def call(x)
      x = MLX::Core.add(norm1.call(attn.call(x)), x)
      MLX::Core.add(norm2.call(ff.call(x)), x)
    end
  end

  class Transformer < MLX::NN::Module
    def initialize(dim, depth, heads, mlp_dim, dropout: 0.0)
      super()
      self.layers = Array.new(depth) { Block.new(dim, heads, mlp_dim, dropout: dropout) }
    end

    def call(x)
      hidden = x
      layers.each do |layer|
        hidden = layer.call(hidden)
      end
      hidden
    end
  end

  class KWT < MLX::NN::Module
    include MLX::DSL::ModelMixin

    attr_reader :num_patches, :dim

    def initialize(
      input_res,
      patch_res,
      num_classes,
      dim:,
      depth:,
      heads:,
      mlp_dim:,
      pool: "mean",
      in_channels: 1,
      dropout: 0.0,
      emb_dropout: 0.0
    )
      super()
      @num_patches = ((input_res[0] / patch_res[0]) * (input_res[1] / patch_res[1])).to_i
      @dim = dim
      @pool = pool

      self.patch_embedding = MLX::NN::Conv2d.new(
        in_channels,
        dim,
        patch_res,
        stride: patch_res
      )
      self.pos_embedding = MLX::Core.truncated_normal(-0.01, 0.01, [@num_patches + 1, dim])
      self.cls_token = MLX::Core.truncated_normal(-0.01, 0.01, [dim])
      self.dropout = MLX::NN::Dropout.new(emb_dropout)
      self.transformer = Transformer.new(dim, depth, heads, mlp_dim, dropout: dropout)
      self.head_norm = MLX::NN::LayerNorm.new(dim)
      self.head_linear = MLX::NN::Linear.new(dim, num_classes)
    end

    def num_params
      MLX::Utils.tree_flatten(parameters).sum { |_k, x| x.size }
    end

    def call(x)
      input = x
      input = MLX::Core.expand_dims(input, -1) if input.ndim != 4
      patches = patch_embedding.call(input)
      patch_count = patches.shape[1] * patches.shape[2]
      patches = MLX::Core.reshape(patches, [patches.shape[0], patch_count, @dim])
      unless patches.shape[1] == @num_patches
        raise "Patch mismatch: expected #{@num_patches}, got #{patches.shape[1]}"
      end

      cls_tokens = MLX::Core.broadcast_to(cls_token, [patches.shape[0], 1, @dim])
      hidden = MLX::Core.concatenate([cls_tokens, patches], 1)
      hidden = MLX::Core.add(hidden, pos_embedding)
      hidden = dropout.call(hidden)
      hidden = transformer.call(hidden)

      pooled = if @pool == "mean"
        MLX::Core.mean(hidden, 1)
      else
        first = MLX::Core.take(hidden, MLX::Core.array([0], MLX::Core.int32), 1)
        MLX::Core.squeeze(first, 1)
      end

      head_linear.call(head_norm.call(pooled))
    end
  end

  module_function

  def parse_kwt_args(kwargs = {})
    options = kwargs.dup
    input_res = options.delete(:input_res) || [98, 40]
    patch_res = options.delete(:patch_res) || [1, 40]
    num_classes = options.delete(:num_classes) || 35
    emb_dropout = options.delete(:emb_dropout) || 0.1
    [input_res, patch_res, num_classes, emb_dropout, options]
  end

  def kwt1(**kwargs)
    input_res, patch_res, num_classes, emb_dropout, rest = parse_kwt_args(kwargs)
    KWT.new(
      input_res,
      patch_res,
      num_classes,
      dim: 64,
      depth: 12,
      heads: 1,
      mlp_dim: 256,
      emb_dropout: emb_dropout,
      **rest
    )
  end

  def kwt2(**kwargs)
    input_res, patch_res, num_classes, emb_dropout, rest = parse_kwt_args(kwargs)
    KWT.new(
      input_res,
      patch_res,
      num_classes,
      dim: 128,
      depth: 12,
      heads: 2,
      mlp_dim: 512,
      emb_dropout: emb_dropout,
      **rest
    )
  end

  def kwt3(**kwargs)
    input_res, patch_res, num_classes, emb_dropout, rest = parse_kwt_args(kwargs)
    KWT.new(
      input_res,
      patch_res,
      num_classes,
      dim: 192,
      depth: 12,
      heads: 3,
      mlp_dim: 768,
      emb_dropout: emb_dropout,
      **rest
    )
  end
end
