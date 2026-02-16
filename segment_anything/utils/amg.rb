# frozen_string_literal: true

require "json"


require "mlx"

module SegmentAnything
  module AMG
    module_function

    def box_xyxy_to_xywh(box)
      arr = box.to_a
      x0, y0, x1, y1 = arr
      [x0, y0, x1 - x0, y1 - y0]
    end

    def batch_iterator(batch_size, array)
      enum_for(:batch_iterator, batch_size, array) unless block_given?
      values = array.to_a
      values.each_slice(batch_size) do |chunk|
        yield MLX::Core.array(chunk)
      end
    end

    def build_point_grid(n_per_side)
      offset = 1.0 / (2 * n_per_side)
      points = []
      n_per_side.times do |y|
        n_per_side.times do |x|
          px = offset + (x.to_f / n_per_side.to_f)
          py = offset + (y.to_f / n_per_side.to_f)
          points << [px, py]
        end
      end
      points
    end

    def build_all_layer_point_grids(n_per_side, n_layers, scale_per_layer)
      (0..n_layers).map do |i|
        n_points = (n_per_side / (scale_per_layer**i)).to_i
        n_points = 1 if n_points < 1
        MLX::Core.array(build_point_grid(n_points), MLX::Core.float32)
      end
    end

    def calculate_stability_score(masks, mask_threshold, threshold_offset)
      high = MLX::Core.greater(masks, mask_threshold + threshold_offset)
      low = MLX::Core.greater(masks, mask_threshold - threshold_offset)
      intersections = MLX::Core.sum(MLX::Core.sum(high.astype(MLX::Core.float32), 1), 1)
      unions = MLX::Core.sum(MLX::Core.sum(low.astype(MLX::Core.float32), 1), 1)
      MLX::Core.divide(intersections, MLX::Core.maximum(unions, 1e-6))
    end

    def batched_mask_to_box(masks)
      mask_arr = masks.to_a
      boxes = mask_arr.map do |mask|
        ys = []
        xs = []
        mask.each_with_index do |row, y|
          row.each_with_index do |v, x|
            if v.to_f > 0.0
              ys << y
              xs << x
            end
          end
        end
        if xs.empty?
          [0, 0, 0, 0]
        else
          [xs.min, ys.min, xs.max, ys.max]
        end
      end
      MLX::Core.array(boxes, MLX::Core.float32)
    end

    def mask_to_rle_mlx(masks)
      mask_arr = masks.to_a
      mask_arr.map do |mask|
        h = mask.length
        w = mask[0].length
        flat = []
        w.times do |x|
          h.times do |y|
            flat << (mask[y][x].to_f > 0.0 ? 1 : 0)
          end
        end

        counts = []
        current = 0
        run = 0
        flat.each do |v|
          if v == current
            run += 1
          else
            counts << run
            run = 1
            current = v
          end
        end
        counts << run
        { "size" => [h, w], "counts" => counts }
      end
    end

    def rle_to_mask(rle)
      h, w = rle.fetch("size")
      counts = rle.fetch("counts")
      flat = []
      parity = 0
      counts.each do |count|
        count.to_i.times { flat << parity }
        parity = 1 - parity
      end
      mask = Array.new(h) { Array.new(w, false) }
      idx = 0
      w.times do |x|
        h.times do |y|
          mask[y][x] = flat[idx].to_i == 1
          idx += 1
        end
      end
      mask
    end

    def area_from_rle(rle)
      counts = rle.fetch("counts")
      counts.each_slice(2).sum do |_off, on|
        on.to_i
      end
    end
  end
end
