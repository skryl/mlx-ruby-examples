# frozen_string_literal: true


require "mlx"

require_relative "config"

module StableDiffusionExample
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
    h_idx = (0...x.shape[1]).step(stride).to_a
    w_idx = (0...x.shape[2]).step(stride).to_a
    h_idx = MLX::Core.array(h_idx, MLX::Core.int32)
    w_idx = MLX::Core.array(w_idx, MLX::Core.int32)
    y = MLX::Core.take(x, h_idx, 1)
    MLX::Core.take(y, w_idx, 2)
  end

  class Autoencoder < MLX::NN::Module
    attr_reader :config, :latent_channels

    def initialize(config)
      super()
      @config = config
      @latent_channels = config.latent_channels_in

      self.encoder_proj = MLX::NN::Linear.new(config.in_channels, config.latent_channels_in)
      self.decoder_proj = MLX::NN::Linear.new(config.latent_channels_in, config.out_channels)
    end

    def encode(image)
      x = image
      x = MLX::Core.expand_dims(x, 0) if x.shape.length == 3
      x = StableDiffusionExample.downsample_stride(x, stride: 8)
      latents = encoder_proj.call(x)
      latents = MLX::Core.multiply(latents, config.scaling_factor)
      [latents, latents]
    end

    def decode(latents)
      x = MLX::Core.divide(latents, config.scaling_factor)
      x = decoder_proj.call(x)
      x = StableDiffusionExample.upsample_nearest(x, scale: 8)
      MLX::Core.tanh(x)
    end
  end
end
