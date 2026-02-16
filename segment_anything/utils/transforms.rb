# frozen_string_literal: true


require "mlx"

module SegmentAnything
  class ResizeLongestSide
    attr_reader :target_length

    def initialize(target_length)
      @target_length = target_length
    end

    def apply_image(image)
      x = image.respond_to?(:shape) ? image : MLX::Core.array(image)
      out_h, out_w = get_preprocess_shape(x.shape[0], x.shape[1], target_length)
      resize_nearest_hwc(x, out_h, out_w)
    end

    def apply_coords(coords, original_size)
      old_h, old_w = original_size
      new_h, new_w = get_preprocess_shape(old_h, old_w, target_length)
      scale = MLX::Core.array([new_w.to_f / old_w.to_f, new_h.to_f / old_h.to_f], MLX::Core.float32)
      MLX::Core.multiply(coords, scale)
    end

    def apply_boxes(boxes, original_size)
      reshaped = MLX::Core.reshape(boxes, [boxes.shape[0], 2, 2])
      scaled = apply_coords(reshaped, original_size)
      MLX::Core.reshape(scaled, [boxes.shape[0], 4])
    end

    def get_preprocess_shape(old_h, old_w, long_side_length)
      scale = long_side_length.to_f / [old_h, old_w].max.to_f
      new_h = (old_h * scale + 0.5).to_i
      new_w = (old_w * scale + 0.5).to_i
      [new_h, new_w]
    end

    private

    def resize_nearest_hwc(image, out_h, out_w)
      in_h = image.shape[0]
      in_w = image.shape[1]
      h_idx = Array.new(out_h) { |i| [(i.to_f * in_h / out_h).floor, in_h - 1].min }
      w_idx = Array.new(out_w) { |i| [(i.to_f * in_w / out_w).floor, in_w - 1].min }

      h_idx = MLX::Core.array(h_idx, MLX::Core.int32)
      w_idx = MLX::Core.array(w_idx, MLX::Core.int32)
      resized = MLX::Core.take(image, h_idx, 0)
      MLX::Core.take(resized, w_idx, 1)
    end
  end
end
