# frozen_string_literal: true

require "optparse"

require_relative "automatic_mask_generator"
require_relative "predictor"
require_relative "sam"

if $PROGRAM_NAME == __FILE__
  options = { seed: 173 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby segment_anything/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  model = SegmentAnything.build_tiny_model(
    image_size: 64,
    patch_size: 8,
    embed_dim: 128,
    depth: 2,
    num_heads: 4,
    prompt_embed_dim: 64
  )

  image = MLX::Core.random_uniform([64, 64, 3], 0.0, 255.0, MLX::Core.float32)
  point_coords = MLX::Core.array([[[20.0, 22.0]]], MLX::Core.float32)
  point_labels = MLX::Core.array([[1]], MLX::Core.int32)

  outputs = model.call(
    [
      {
        "image" => image,
        "original_size" => [64, 64],
        "point_coords" => point_coords,
        "point_labels" => point_labels
      }
    ],
    multimask_output: true
  )

  raise "SAM call should return one output" unless outputs.length == 1
  out = outputs[0]
  masks = out.fetch("masks")
  iou = out.fetch("iou_predictions")
  low_res = out.fetch("low_res_logits")
  MLX::Core.eval(masks, iou, low_res)

  unless masks.shape[0] == 1 && masks.shape[1] == 64 && masks.shape[2] == 64 && masks.shape[3] == 3
    raise "Unexpected SAM output mask shape #{masks.shape.inspect}"
  end
  raise "Unexpected IoU shape #{iou.shape.inspect}" unless iou.shape == [1, 3]

  predictor = SegmentAnything::SamPredictor.new(model)
  predictor.set_image(image)
  pmasks, piou, plow = predictor.predict(
    point_coords: MLX::Core.array([[20.0, 22.0]], MLX::Core.float32),
    point_labels: MLX::Core.array([1], MLX::Core.int32),
    multimask_output: false,
    return_logits: true
  )
  MLX::Core.eval(pmasks, piou, plow)

  unless pmasks.shape[0] == 1 && pmasks.shape[1] == 64 && pmasks.shape[2] == 64
    raise "Predictor mask shape mismatch: #{pmasks.shape.inspect}"
  end
  raise "Predictor IoU shape mismatch: #{piou.shape.inspect}" unless piou.shape == [1, 1]

  generator = SegmentAnything::SamAutomaticMaskGenerator.new(
    model: model,
    points_per_side: 4,
    points_per_batch: 4,
    pred_iou_thresh: -1.0,
    stability_score_thresh: -1.0,
    output_mode: "binary_mask"
  )
  anns = generator.generate(image)
  raise "Mask generator produced no annotations" if anns.empty?

  first = anns.first
  required = %w[segmentation area bbox predicted_iou point_coords stability_score crop_box]
  missing = required.reject { |k| first.key?(k) }
  raise "Missing annotation keys: #{missing.join(', ')}" unless missing.empty?

  puts "Tests pass :)"
end
