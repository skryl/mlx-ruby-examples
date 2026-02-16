# frozen_string_literal: true


require "mlx"
require "mlx/dsl"

module FluxExample
  class AutoEncoderParams
    include MLX::DSL::ConfigSchema

    field :resolution, Integer, default: 256
    field :in_channels, Integer, default: 3
    field :ch, Integer, default: 128
    field :out_ch, Integer, default: 3
    field :ch_mult, Array, default: [1, 2, 4, 4]
    field :num_res_blocks, Integer, default: 2
    field :z_channels, Integer, default: 16
    field :scale_factor, [Integer, Float], default: 0.3611
    field :shift_factor, [Integer, Float], default: 0.1159
  end

  module AutoencoderOps
    module_function

    def upsample_nearest(x, scale: 8)
      batch, height, width, channels = x.shape

      x = MLX::Core.expand_dims(x, 2)
      x = MLX::Core.concatenate(Array.new(scale, x), 2)
      x = MLX::Core.reshape(x, [batch, height * scale, width, channels])

      x = MLX::Core.expand_dims(x, 3)
      x = MLX::Core.concatenate(Array.new(scale, x), 3)
      MLX::Core.reshape(x, [batch, height * scale, width * scale, channels])
    end

    def downsample_stride(x, stride: 8)
      h_idx = MLX::Core.arange(0, x.shape[1], stride, MLX::Core.int32)
      w_idx = MLX::Core.arange(0, x.shape[2], stride, MLX::Core.int32)
      y = MLX::Core.take(x, h_idx, 1)
      MLX::Core.take(y, w_idx, 2)
    end
  end

  class AutoEncoder < MLX::NN::Module
    attr_reader :scale_factor, :shift_factor

    def initialize(params)
      super()
      self.encoder_proj = MLX::NN::Linear.new(params.in_channels, params.z_channels)
      self.decoder_proj = MLX::NN::Linear.new(params.z_channels, params.out_ch)
      @scale_factor = params.scale_factor
      @shift_factor = params.shift_factor
    end

    def sanitize(weights)
      weights.each_with_object({}) do |(key, value), out|
        w = value
        if w.shape.length == 4
          w = MLX::Core.transpose(w, [0, 2, 3, 1])
        end
        out[key.to_s] = w
      end
    end

    def encode(x)
      z = AutoencoderOps.downsample_stride(x, stride: 8)
      z = encoder_proj.call(z)
      MLX::Core.multiply(scale_factor, MLX::Core.subtract(z, shift_factor))
    end

    def decode(z)
      z = MLX::Core.add(MLX::Core.divide(z, scale_factor), shift_factor)
      y = decoder_proj.call(z)
      AutoencoderOps.upsample_nearest(y, scale: 8)
    end

    def call(x)
      decode(encode(x))
    end
  end
end
