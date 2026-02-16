# frozen_string_literal: true

require_relative "sam"
require_relative "utils/transforms"

module SegmentAnything
  class SamPredictor
    attr_reader :model, :transform, :features, :original_size, :input_size
    attr_accessor :is_image_set

    def initialize(sam_model)
      @model = sam_model
      @transform = ResizeLongestSide.new(sam_model.vision_encoder.img_size)
      reset_image
    end

    def set_image(image, image_format: "RGB")
      reset_image
      unless %w[RGB BGR].include?(image_format)
        raise ArgumentError, "image_format must be one of RGB/BGR"
      end

      x = image.respond_to?(:shape) ? image : MLX::Core.array(image)
      if image_format != model.image_format
        arr = x.to_a
        arr.each do |row|
          row.each do |pix|
            pix[0], pix[2] = pix[2], pix[0]
          end
        end
        x = MLX::Core.array(arr, x.dtype)
      end

      input_image = transform.apply_image(x)
      input_image = MLX::Core.expand_dims(input_image, 0)

      @original_size = [x.shape[0], x.shape[1]]
      @input_size = [input_image.shape[1], input_image.shape[2]]

      input_image = model.preprocess(input_image)
      @features = model.vision_encoder.call(input_image)
      self.is_image_set = true
    end

    def predict(
      point_coords: nil,
      point_labels: nil,
      box: nil,
      mask_input: nil,
      multimask_output: true,
      return_logits: false
    )
      unless is_image_set
        raise RuntimeError, "An image must be set with .set_image(...) before prediction"
      end

      points = nil
      unless point_coords.nil?
        raise ArgumentError, "point_labels must be provided with point_coords" if point_labels.nil?

        point_coords = transform.apply_coords(point_coords, original_size)
        point_coords = MLX::Core.expand_dims(point_coords, 0) if point_coords.shape.length == 2
        point_labels = MLX::Core.expand_dims(point_labels, 0) if point_labels.shape.length == 1
        points = [point_coords, point_labels]
      end

      unless box.nil?
        box = MLX::Core.expand_dims(box, 0) if box.shape.length == 1
        box = transform.apply_boxes(box, original_size)
      end

      sparse_embeddings, dense_embeddings = model.prompt_encoder.call(
        points: points,
        boxes: box,
        masks: mask_input,
        pe_layer: model.shared_image_embedding
      )

      low_res_masks, iou_predictions = model.mask_decoder.call(
        image_embeddings: features,
        image_pe: model.shared_image_embedding.call(model.prompt_encoder.image_embedding_size),
        sparse_prompt_embeddings: sparse_embeddings,
        dense_prompt_embeddings: dense_embeddings,
        multimask_output: multimask_output
      )

      masks = model.postprocess_masks(low_res_masks, input_size: input_size, original_size: original_size)
      masks = MLX::Core.greater(masks, model.mask_threshold) unless return_logits

      [masks, iou_predictions, low_res_masks]
    end

    def get_image_embedding
      unless is_image_set
        raise RuntimeError, "An image must be set before getting embeddings"
      end

      features
    end

    def reset_image
      self.is_image_set = false
      @features = nil
      @original_size = nil
      @input_size = nil
    end
  end
end
