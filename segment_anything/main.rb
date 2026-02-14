# frozen_string_literal: true

require "json"
require "optparse"
require "fileutils"

require_relative "automatic_mask_generator"
require_relative "sam"

if $PROGRAM_NAME == __FILE__
  options = {
    input_npz: nil,
    output: "segment_anything/output",
    model: nil,
    output_mode: "binary_mask",
    points_per_side: 8,
    points_per_batch: 32,
    pred_iou_thresh: 0.0,
    stability_score_thresh: 0.0,
    seed: 123
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby segment_anything/main.rb [options]"
    opts.on("--input-npz PATH", String, "Optional NPZ with key 'image' (H,W,3)") { |v| options[:input_npz] = v }
    opts.on("--output PATH", String, "Output directory") { |v| options[:output] = v }
    opts.on("--model PATH", String, "Optional converted SAM model directory") { |v| options[:model] = v }
    opts.on("--output-mode MODE", String, "binary_mask|uncompressed_rle|coco_rle") { |v| options[:output_mode] = v }
    opts.on("--points-per-side N", Integer, "Grid points per side") { |v| options[:points_per_side] = v }
    opts.on("--points-per-batch N", Integer, "Points per predictor batch") { |v| options[:points_per_batch] = v }
    opts.on("--pred-iou-thresh N", Float, "Predicted IoU threshold") { |v| options[:pred_iou_thresh] = v }
    opts.on("--stability-score-thresh N", Float, "Stability score threshold") { |v| options[:stability_score_thresh] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  model = if !options[:model].nil? && Dir.exist?(options[:model])
    SegmentAnything.load(options[:model])
  else
    SegmentAnything.build_tiny_model
  end

  image = if !options[:input_npz].nil? && File.exist?(options[:input_npz])
    payload = MLX::Core.load(options[:input_npz])
    payload["image"] || payload[:image] || raise("Missing 'image' in #{options[:input_npz]}")
  else
    MLX::Core.random_uniform([64, 64, 3], 0.0, 255.0, MLX::Core.float32)
  end

  generator = SegmentAnything::SamAutomaticMaskGenerator.new(
    model: model,
    points_per_side: options[:points_per_side],
    points_per_batch: options[:points_per_batch],
    pred_iou_thresh: options[:pred_iou_thresh],
    stability_score_thresh: options[:stability_score_thresh],
    output_mode: options[:output_mode]
  )

  masks = generator.generate(image)

  FileUtils.mkdir_p(options[:output])
  out_file = File.join(options[:output], "masks.json")
  File.binwrite(out_file, JSON.pretty_generate(masks))

  puts "generated_masks=#{masks.length}"
  puts "output=#{out_file}"
end
