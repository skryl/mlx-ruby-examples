# frozen_string_literal: true

require "ostruct"


require "mlx"

module FluxExample
  module Layers
    module_function

    def rope(pos, dim, theta)
      scale = MLX::Core.divide(MLX::Core.arange(0, dim, 2, MLX::Core.float32), dim.to_f)
      omega = MLX::Core.divide(1.0, MLX::Core.power(theta.to_f, scale))
      x = MLX::Core.multiply(MLX::Core.expand_dims(pos, -1), omega)
      cosx = MLX::Core.cos(x)
      sinx = MLX::Core.sin(x)
      pe = MLX::Core.stack([cosx, MLX::Core.multiply(-1.0, sinx), sinx, cosx], -1)
      MLX::Core.reshape(pe, pe.shape[0...-1] + [2, 2])
    end

    def timestep_embedding(t, dim, max_period: 10_000, time_factor: 1000.0)
      half = dim / 2
      freqs = MLX::Core.divide(MLX::Core.arange(0, half, 1, MLX::Core.float32), half.to_f)
      freqs = MLX::Core.multiply(freqs, -Math.log(max_period))
      freqs = MLX::Core.exp(freqs)

      x = MLX::Core.multiply(MLX::Core.expand_dims(MLX::Core.multiply(time_factor, t), 1), MLX::Core.expand_dims(freqs, 0))
      out = MLX::Core.concatenate([MLX::Core.cos(x), MLX::Core.sin(x)], 1)
      out.astype(t.dtype)
    end
  end

  class EmbedND < MLX::NN::Module
    def initialize(dim:, theta:, axes_dim:)
      super()
      @dim = dim
      @theta = theta
      @axes_dim = axes_dim
    end

    def call(ids)
      n_axes = ids.shape[-1]
      pes = []
      n_axes.times do |i|
        coord = MLX::Core.squeeze(MLX::Core.take(ids, MLX::Core.array([i], MLX::Core.int32), -1), -1)
        axis_dim = @axes_dim[i] || @axes_dim[-1]
        pes << Layers.rope(coord, axis_dim, @theta)
      end
      pe = MLX::Core.concatenate(pes, -3)
      MLX::Core.expand_dims(pe, 1)
    end
  end

  class MLPEmbedder < MLX::NN::Module
    def initialize(in_dim:, hidden_dim:)
      super()
      self.in_layer = MLX::NN::Linear.new(in_dim, hidden_dim, bias: true)
      self.out_layer = MLX::NN::Linear.new(hidden_dim, hidden_dim, bias: true)
    end

    def call(x)
      out_layer.call(MLX::NN.silu(in_layer.call(x)))
    end
  end

  class QKNorm < MLX::NN::Module
    def initialize(dim:)
      super()
      self.query_norm = MLX::NN::RMSNorm.new(dim)
      self.key_norm = MLX::NN::RMSNorm.new(dim)
    end

    def call(q, k)
      [query_norm.call(q), key_norm.call(k)]
    end
  end

  class SelfAttention < MLX::NN::Module
    attr_reader :num_heads

    def initialize(dim:, num_heads: 8, qkv_bias: false)
      super()
      @num_heads = num_heads
      head_dim = dim / num_heads
      @head_dim = head_dim
      @scale = head_dim**-0.5

      self.qkv = MLX::NN::Linear.new(dim, dim * 3, bias: qkv_bias)
      self.norm = QKNorm.new(dim: head_dim)
      self.proj = MLX::NN::Linear.new(dim, dim)
    end

    def call(x, _pe = nil)
      h = num_heads
      b, l, _d = x.shape
      qkv_val = qkv.call(x)
      q, k, v = MLX::Core.split(qkv_val, 3, -1)
      q = MLX::Core.transpose(MLX::Core.reshape(q, [b, l, h, @head_dim]), [0, 2, 1, 3])
      k = MLX::Core.transpose(MLX::Core.reshape(k, [b, l, h, @head_dim]), [0, 2, 1, 3])
      v = MLX::Core.transpose(MLX::Core.reshape(v, [b, l, h, @head_dim]), [0, 2, 1, 3])
      q, k = norm.call(q, k)
      y = MLX::Core.scaled_dot_product_attention(q, k, v, @scale, nil)
      y = MLX::Core.transpose(y, [0, 2, 1, 3])
      y = MLX::Core.reshape(y, [b, l, h * @head_dim])
      proj.call(y)
    end
  end

  ModulationOut = Struct.new(:shift, :scale, :gate)

  class Modulation < MLX::NN::Module
    def initialize(dim:, double:)
      super()
      @is_double = double
      @multiplier = double ? 6 : 3
      self.lin = MLX::NN::Linear.new(dim, @multiplier * dim, bias: true)
    end

    def call(x)
      y = lin.call(MLX::NN.silu(x))
      parts = MLX::Core.split(MLX::Core.expand_dims(y, 1), @multiplier, -1)
      mod1 = ModulationOut.new(parts[0], parts[1], parts[2])
      mod2 = @is_double ? ModulationOut.new(parts[3], parts[4], parts[5]) : nil
      [mod1, mod2]
    end
  end

  class DoubleStreamBlock < MLX::NN::Module
    attr_accessor :num_heads, :hidden_size, :sharding_group

    def initialize(hidden_size, num_heads, mlp_ratio:, qkv_bias: false)
      super()
      @num_heads = num_heads
      @hidden_size = hidden_size
      mlp_hidden_dim = (hidden_size * mlp_ratio).to_i

      self.img_mod = Modulation.new(dim: hidden_size, double: true)
      self.img_norm1 = MLX::NN::LayerNorm.new(hidden_size, affine: false, eps: 1e-6)
      self.img_attn = SelfAttention.new(dim: hidden_size, num_heads: num_heads, qkv_bias: qkv_bias)
      self.img_norm2 = MLX::NN::LayerNorm.new(hidden_size, affine: false, eps: 1e-6)
      self.img_mlp_1 = MLX::NN::Linear.new(hidden_size, mlp_hidden_dim, bias: true)
      self.img_mlp_2 = MLX::NN::Linear.new(mlp_hidden_dim, hidden_size, bias: true)

      self.txt_mod = Modulation.new(dim: hidden_size, double: true)
      self.txt_norm1 = MLX::NN::LayerNorm.new(hidden_size, affine: false, eps: 1e-6)
      self.txt_attn = SelfAttention.new(dim: hidden_size, num_heads: num_heads, qkv_bias: qkv_bias)
      self.txt_norm2 = MLX::NN::LayerNorm.new(hidden_size, affine: false, eps: 1e-6)
      self.txt_mlp_1 = MLX::NN::Linear.new(hidden_size, mlp_hidden_dim, bias: true)
      self.txt_mlp_2 = MLX::NN::Linear.new(mlp_hidden_dim, hidden_size, bias: true)

      @sharding_group = nil
    end

    def call(img:, txt:, vec:, pe: nil)
      _ = pe
      img_mod1, img_mod2 = img_mod.call(vec)
      txt_mod1, txt_mod2 = txt_mod.call(vec)

      img_modulated = MLX::Core.add(
        MLX::Core.multiply(MLX::Core.add(1.0, img_mod1.scale), img_norm1.call(img)),
        img_mod1.shift
      )
      txt_modulated = MLX::Core.add(
        MLX::Core.multiply(MLX::Core.add(1.0, txt_mod1.scale), txt_norm1.call(txt)),
        txt_mod1.shift
      )

      img_attn_out = img_attn.call(img_modulated)
      txt_attn_out = txt_attn.call(txt_modulated)

      img = MLX::Core.add(img, MLX::Core.multiply(img_mod1.gate, img_attn_out))
      txt = MLX::Core.add(txt, MLX::Core.multiply(txt_mod1.gate, txt_attn_out))

      img_mlp_in = MLX::Core.add(
        MLX::Core.multiply(MLX::Core.add(1.0, img_mod2.scale), img_norm2.call(img)),
        img_mod2.shift
      )
      img_mlp_out = img_mlp_2.call(MLX::NN.gelu(img_mlp_1.call(img_mlp_in)))

      txt_mlp_in = MLX::Core.add(
        MLX::Core.multiply(MLX::Core.add(1.0, txt_mod2.scale), txt_norm2.call(txt)),
        txt_mod2.shift
      )
      txt_mlp_out = txt_mlp_2.call(MLX::NN.gelu(txt_mlp_1.call(txt_mlp_in)))

      img = MLX::Core.add(img, MLX::Core.multiply(img_mod2.gate, img_mlp_out))
      txt = MLX::Core.add(txt, MLX::Core.multiply(txt_mod2.gate, txt_mlp_out))

      [img, txt]
    end
  end

  class SingleStreamBlock < MLX::NN::Module
    attr_accessor :num_heads, :hidden_size

    def initialize(hidden_size, num_heads, mlp_ratio: 4.0, qk_scale: nil)
      super()
      @hidden_size = hidden_size
      @num_heads = num_heads
      @head_dim = hidden_size / num_heads
      @scale = qk_scale || @head_dim**-0.5

      @mlp_hidden_dim = (hidden_size * mlp_ratio).to_i

      self.linear1 = MLX::NN::Linear.new(hidden_size, hidden_size * 3 + @mlp_hidden_dim)
      self.linear2 = MLX::NN::Linear.new(hidden_size + @mlp_hidden_dim, hidden_size)
      self.norm = QKNorm.new(dim: @head_dim)
      self.pre_norm = MLX::NN::LayerNorm.new(hidden_size, affine: false, eps: 1e-6)
      self.modulation = Modulation.new(dim: hidden_size, double: false)
    end

    def call(x, vec:, pe: nil)
      _ = pe
      b, l, _ = x.shape
      mod, = modulation.call(vec)
      x_mod = MLX::Core.add(
        MLX::Core.multiply(MLX::Core.add(1.0, mod.scale), pre_norm.call(x)),
        mod.shift
      )

      split_points = [@hidden_size, 2 * @hidden_size, 3 * @hidden_size]
      q, k, v, mlp = MLX::Core.split(linear1.call(x_mod), split_points, -1)

      q = MLX::Core.transpose(MLX::Core.reshape(q, [b, l, num_heads, @head_dim]), [0, 2, 1, 3])
      k = MLX::Core.transpose(MLX::Core.reshape(k, [b, l, num_heads, @head_dim]), [0, 2, 1, 3])
      v = MLX::Core.transpose(MLX::Core.reshape(v, [b, l, num_heads, @head_dim]), [0, 2, 1, 3])
      q, k = norm.call(q, k)
      attn_out = MLX::Core.scaled_dot_product_attention(q, k, v, @scale, nil)
      attn_out = MLX::Core.transpose(attn_out, [0, 2, 1, 3])
      attn_out = MLX::Core.reshape(attn_out, [b, l, hidden_size])

      y = MLX::Core.concatenate([attn_out, MLX::NN.gelu(mlp)], 2)
      y = linear2.call(y)
      MLX::Core.add(x, MLX::Core.multiply(mod.gate, y))
    end
  end

  class LastLayer < MLX::NN::Module
    def initialize(hidden_size, patch_size, out_channels)
      super()
      self.norm_final = MLX::NN::LayerNorm.new(hidden_size, affine: false, eps: 1e-6)
      self.linear = MLX::NN::Linear.new(hidden_size, patch_size * patch_size * out_channels, bias: true)
      self.mod1 = MLX::NN::Linear.new(hidden_size, hidden_size, bias: true)
      self.mod2 = MLX::NN::Linear.new(hidden_size, hidden_size, bias: true)
    end

    def call(x, vec)
      h = MLX::NN.silu(vec)
      shift = mod1.call(h)
      scale = mod2.call(h)
      y = MLX::Core.add(
        MLX::Core.multiply(MLX::Core.add(1.0, MLX::Core.expand_dims(scale, 1)), norm_final.call(x)),
        MLX::Core.expand_dims(shift, 1)
      )
      linear.call(y)
    end
  end
end
