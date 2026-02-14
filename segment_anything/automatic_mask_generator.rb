# frozen_string_literal: true

require_relative "predictor"
require_relative "utils/amg"

module SegmentAnything
  class SamAutomaticMaskGenerator
    def initialize(
      model:,
      points_per_side: 16,
      points_per_batch: 64,
      pred_iou_thresh: 0.88,
      stability_score_thresh: 0.95,
      stability_score_offset: 1.0,
      box_nms_thresh: 0.7,
      crop_n_layers: 0,
      crop_n_points_downscale_factor: 1,
      output_mode: "binary_mask",
      **_kwargs
    )
      @predictor = SamPredictor.new(model)
      @points_per_batch = points_per_batch
      @pred_iou_thresh = pred_iou_thresh
      @stability_score_thresh = stability_score_thresh
      @stability_score_offset = stability_score_offset
      @box_nms_thresh = box_nms_thresh
      @output_mode = output_mode

      @point_grids = AMG.build_all_layer_point_grids(
        points_per_side,
        crop_n_layers,
        crop_n_points_downscale_factor
      )
    end

    def generate(image)
      @predictor.set_image(image)
      h, w = image.shape[0], image.shape[1]

      points = @point_grids[0]
      scale = MLX::Core.array([w.to_f, h.to_f], MLX::Core.float32)
      points = MLX::Core.multiply(points, scale)

      anns = []
      AMG.batch_iterator(@points_per_batch, points) do |batch_points|
        labels = MLX::Core.ones([batch_points.shape[0]], MLX::Core.int32)
        masks, iou_preds, low_res = @predictor.predict(
          point_coords: batch_points,
          point_labels: labels,
          multimask_output: false,
          return_logits: true
        )

        masks = MLX::Core.squeeze(masks, 3)
        low_res = MLX::Core.squeeze(low_res, 3)
        iou_preds = MLX::Core.squeeze(iou_preds, 1)
        stability = AMG.calculate_stability_score(
          low_res,
          @predictor.model.mask_threshold,
          @stability_score_offset
        )

        batch_points_a = batch_points.to_a
        iou_a = iou_preds.to_a
        stability_a = stability.to_a

        batch_points.shape[0].times do |i|
          next if iou_a[i].to_f < @pred_iou_thresh
          next if stability_a[i].to_f < @stability_score_thresh

          mask_i = MLX::Core.squeeze(
            MLX::Core.take(masks, MLX::Core.array([i], MLX::Core.int32), 0),
            0
          )
          rle = AMG.mask_to_rle_mlx(MLX::Core.expand_dims(mask_i, 0))[0]
          box = AMG.batched_mask_to_box(MLX::Core.expand_dims(mask_i, 0)).to_a[0]

          segmentation = case @output_mode
          when "binary_mask"
            AMG.rle_to_mask(rle)
          when "uncompressed_rle", "coco_rle"
            rle
          else
            raise ArgumentError, "Unknown output_mode #{@output_mode}"
          end

          anns << {
            "segmentation" => segmentation,
            "area" => AMG.area_from_rle(rle),
            "bbox" => AMG.box_xyxy_to_xywh(MLX::Core.array(box)).map(&:to_f),
            "predicted_iou" => iou_a[i].to_f,
            "point_coords" => [batch_points_a[i]],
            "stability_score" => stability_a[i].to_f,
            "crop_box" => [0.0, 0.0, w.to_f, h.to_f]
          }
        end
      end

      anns
    ensure
      @predictor.reset_image
    end
  end
end
