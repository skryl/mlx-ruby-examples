# frozen_string_literal: true


require "mlx"

module ClipExample
  class ImageProcessor
    DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073].freeze
    DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711].freeze

    def initialize(image_size: 224, mean: DEFAULT_MEAN, std: DEFAULT_STD)
      @image_size = image_size
      @mean = MLX::Core.reshape(MLX::Core.array(mean, MLX::Core.float32), [1, 1, 1, 3])
      @std = MLX::Core.reshape(MLX::Core.array(std, MLX::Core.float32), [1, 1, 1, 3])
    end

    def call(images)
      x = images
      x = MLX::Core.expand_dims(x, 0) if x.ndim == 3
      unless x.shape[-1] == 3
        raise ArgumentError, "Expected channel-last images with 3 channels, got shape #{x.shape.inspect}"
      end

      x = resize_nearest(x, @image_size, @image_size) if x.shape[1] != @image_size || x.shape[2] != @image_size
      x = MLX::Core.divide(x.astype(MLX::Core.float32), 255.0)
      MLX::Core.divide(MLX::Core.subtract(x, @mean), @std)
    end

    private

    def resize_nearest(images, out_h, out_w)
      in_h = images.shape[1]
      in_w = images.shape[2]
      h_idx = MLX::Core.array((0...out_h).map { |i| (i * in_h) / out_h }, MLX::Core.int32)
      w_idx = MLX::Core.array((0...out_w).map { |i| (i * in_w) / out_w }, MLX::Core.int32)
      resized = MLX::Core.take(images, h_idx, 1)
      MLX::Core.take(resized, w_idx, 2)
    end
  end
end
