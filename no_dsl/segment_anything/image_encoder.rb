# frozen_string_literal: true

require_relative "common"

module SegmentAnything
  class PatchEmbed < MLX::NN::Module
    def initialize(kernel_size:, stride:, padding:, in_chans:, embed_dim:)
      super()
      self.projection = MLX::NN::Conv2d.new(
        in_chans,
        embed_dim,
        kernel_size,
        stride: stride,
        padding: padding
      )
    end

    def call(x)
      projection.call(x)
    end
  end

  class ImageAttention < MLX::NN::Module
    def initialize(dim:, num_heads:, qkv_bias: true)
      super()
      @num_heads = num_heads
      @head_dim = dim / num_heads
      @scale = @head_dim**-0.5

      self.qkv = MLX::NN::Linear.new(dim, dim * 3, bias: qkv_bias)
      self.proj = MLX::NN::Linear.new(dim, dim)
    end

    def call(x)
      b, h, w, dim = x.shape
      hw = h * w

      qkv = self.qkv.call(x)
      qkv = MLX::Core.reshape(qkv, [b, hw, 3, @num_heads, @head_dim])
      qkv = MLX::Core.transpose(qkv, [2, 0, 3, 1, 4])
      q = MLX::Core.take(qkv, MLX::Core.array([0], MLX::Core.int32), 0)
      k = MLX::Core.take(qkv, MLX::Core.array([1], MLX::Core.int32), 0)
      v = MLX::Core.take(qkv, MLX::Core.array([2], MLX::Core.int32), 0)
      q = MLX::Core.squeeze(q, 0)
      k = MLX::Core.squeeze(k, 0)
      v = MLX::Core.squeeze(v, 0)

      attn = MLX::Core.matmul(
        MLX::Core.multiply(q, @scale),
        MLX::Core.transpose(k, [0, 1, 3, 2])
      )
      attn = MLX::Core.softmax(attn, -1)
      out = MLX::Core.matmul(attn, v)
      out = MLX::Core.transpose(out, [0, 2, 1, 3])
      out = MLX::Core.reshape(out, [b, h, w, dim])
      proj.call(out)
    end
  end

  class Block < MLX::NN::Module
    def initialize(dim:, num_heads:, mlp_ratio: 4.0, qkv_bias: true, norm_eps: 1e-6)
      super()
      self.layer_norm1 = MLX::NN::LayerNorm.new(dim, eps: norm_eps)
      self.attn = ImageAttention.new(dim: dim, num_heads: num_heads, qkv_bias: qkv_bias)
      self.layer_norm2 = MLX::NN::LayerNorm.new(dim, eps: norm_eps)
      self.mlp = MLPBlock.new(embedding_dim: dim, mlp_dim: (dim * mlp_ratio).to_i)
    end

    def call(x)
      h = MLX::Core.add(x, attn.call(layer_norm1.call(x)))
      MLX::Core.add(h, mlp.call(layer_norm2.call(h)))
    end
  end

  class Neck < MLX::NN::Module
    def initialize(embed_dim, out_chans)
      super()
      self.conv1 = MLX::NN::Conv2d.new(embed_dim, out_chans, 1, bias: false)
      self.layer_norm1 = LayerNorm2d.new(out_chans)
      self.conv2 = MLX::NN::Conv2d.new(out_chans, out_chans, 3, padding: 1, bias: false)
      self.layer_norm2 = LayerNorm2d.new(out_chans)
    end

    def call(x)
      layer_norm2.call(conv2.call(layer_norm1.call(conv1.call(x))))
    end
  end

  class ImageEncoderViT < MLX::NN::Module
    attr_reader :img_size

    def initialize(
      img_size: 1024,
      patch_size: 16,
      in_chans: 3,
      embed_dim: 768,
      depth: 12,
      num_heads: 12,
      mlp_ratio: 4.0,
      out_chans: 256,
      qkv_bias: true,
      use_abs_pos: true,
      **_kwargs
    )
      super()
      @img_size = img_size

      self.patch_embed = PatchEmbed.new(
        kernel_size: patch_size,
        stride: patch_size,
        padding: 0,
        in_chans: in_chans,
        embed_dim: embed_dim
      )

      if use_abs_pos
        side = img_size / patch_size
        self.pos_embed = MLX::Core.zeros([1, side, side, embed_dim])
      else
        self.pos_embed = nil
      end

      self.layers = Array.new(depth) do
        Block.new(
          dim: embed_dim,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          qkv_bias: qkv_bias
        )
      end

      self.neck = Neck.new(embed_dim, out_chans)
    end

    def call(x)
      x = patch_embed.call(x)
      x = MLX::Core.add(x, pos_embed) unless pos_embed.nil?
      layers.each { |blk| x = blk.call(x) }
      neck.call(x)
    end
  end
end
