# frozen_string_literal: true


require "mlx"

module CvaeExample
  module_function

  def upsample_nearest(x, scale: 2)
    batch, height, width, channels = x.shape

    x = MLX::Core.expand_dims(x, 2)
    x = MLX::Core.concatenate(Array.new(scale, x), 2)
    x = MLX::Core.reshape(x, [batch, height * scale, width, channels])

    x = MLX::Core.expand_dims(x, 3)
    x = MLX::Core.concatenate(Array.new(scale, x), 3)
    MLX::Core.reshape(x, [batch, height * scale, width * scale, channels])
  end

  class UpsamplingConv2d < MLX::NN::Module
    def initialize(in_channels, out_channels, kernel_size, stride:, padding:)
      super()
      self.conv = MLX::NN::Conv2d.new(
        in_channels,
        out_channels,
        kernel_size,
        stride: stride,
        padding: padding
      )
    end

    def call(x)
      conv.call(CvaeExample.upsample_nearest(x))
    end
  end

  class Encoder < MLX::NN::Module
    def initialize(num_latent_dims, image_shape, max_num_filters)
      super()

      num_filters_1 = max_num_filters / 4
      num_filters_2 = max_num_filters / 2
      num_filters_3 = max_num_filters

      self.conv1 = MLX::NN::Conv2d.new(image_shape[-1], num_filters_1, 3, stride: 2, padding: 1)
      self.conv2 = MLX::NN::Conv2d.new(num_filters_1, num_filters_2, 3, stride: 2, padding: 1)
      self.conv3 = MLX::NN::Conv2d.new(num_filters_2, num_filters_3, 3, stride: 2, padding: 1)

      self.bn1 = MLX::NN::BatchNorm.new(num_filters_1)
      self.bn2 = MLX::NN::BatchNorm.new(num_filters_2)
      self.bn3 = MLX::NN::BatchNorm.new(num_filters_3)

      output_shape = [num_filters_3] + image_shape[0...-1].map { |dim| dim / 8 }
      flattened_dim = output_shape.reduce(1, :*)

      self.proj_mu = MLX::NN::Linear.new(flattened_dim, num_latent_dims)
      self.proj_log_var = MLX::NN::Linear.new(flattened_dim, num_latent_dims)
    end

    def call(x)
      x = MLX::NN.leaky_relu(bn1.call(conv1.call(x)))
      x = MLX::NN.leaky_relu(bn2.call(conv2.call(x)))
      x = MLX::NN.leaky_relu(bn3.call(conv3.call(x)))
      x = MLX::Core.flatten(x, 1)

      mu = proj_mu.call(x)
      logvar = proj_log_var.call(x)
      sigma = MLX::Core.exp(MLX::Core.multiply(logvar, 0.5))
      eps = MLX::Core.normal(sigma.shape)
      z = MLX::Core.add(MLX::Core.multiply(eps, sigma), mu)
      [z, mu, logvar]
    end
  end

  class Decoder < MLX::NN::Module
    def initialize(num_latent_dims, image_shape, max_num_filters)
      super()
      @max_num_filters = max_num_filters
      num_img_channels = image_shape[-1]

      num_filters_1 = max_num_filters
      num_filters_2 = max_num_filters / 2
      num_filters_3 = max_num_filters / 4

      @input_shape = image_shape[0...-1].map { |dim| dim / 8 } + [num_filters_1]
      flattened_dim = @input_shape.reduce(1, :*)

      self.lin1 = MLX::NN::Linear.new(num_latent_dims, flattened_dim)
      self.upconv1 = UpsamplingConv2d.new(num_filters_1, num_filters_2, 3, stride: 1, padding: 1)
      self.upconv2 = UpsamplingConv2d.new(num_filters_2, num_filters_3, 3, stride: 1, padding: 1)
      self.upconv3 = UpsamplingConv2d.new(num_filters_3, num_img_channels, 3, stride: 1, padding: 1)

      self.bn1 = MLX::NN::BatchNorm.new(num_filters_2)
      self.bn2 = MLX::NN::BatchNorm.new(num_filters_3)
    end

    def call(z)
      x = lin1.call(z)
      x = MLX::Core.reshape(x, [z.shape[0], @input_shape[0], @input_shape[1], @max_num_filters])
      x = MLX::NN.leaky_relu(bn1.call(upconv1.call(x)))
      x = MLX::NN.leaky_relu(bn2.call(upconv2.call(x)))
      MLX::Core.sigmoid(upconv3.call(x))
    end
  end

  class CVAE < MLX::NN::Module
    include MLX::DSL::ModelMixin

    attr_reader :num_latent_dims

    def initialize(num_latent_dims, input_shape, max_num_filters)
      super()
      @num_latent_dims = num_latent_dims
      self.encoder = Encoder.new(num_latent_dims, input_shape, max_num_filters)
      self.decoder = Decoder.new(num_latent_dims, input_shape, max_num_filters)
    end

    def call(x)
      z, mu, logvar = encoder.call(x)
      [decode(z), mu, logvar]
    end

    def encode(x)
      z, = encoder.call(x)
      z
    end

    def decode(z)
      decoder.call(z)
    end
  end
end
