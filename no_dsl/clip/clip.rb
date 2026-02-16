# frozen_string_literal: true

require "optparse"

require_relative "image_processor"
require_relative "model"
require_relative "tokenizer"

module ClipExample
  module Runner
    module_function

    def run(options)
      tokenizer = SimpleTokenizer.new(max_length: options[:context_length])
      processor = ImageProcessor.new(image_size: options[:image_size])
      model = CLIPModel.new(
        vocab_size: tokenizer.vocab_size,
        text_width: options[:text_width],
        vision_width: options[:vision_width],
        embed_dim: options[:embed_dim],
        image_size: options[:image_size],
        patch_size: options[:patch_size],
        max_length: options[:context_length]
      )

      texts = options[:texts]
      if texts.empty?
        texts = ["a photo of a cat", "a photo of a dog"]
      end

      images = if options[:images_npz]
        data = MLX::Core.load(options[:images_npz]).to_a.to_h
        key, value = data.first
        puts "Loaded images from key #{key}"
        value
      else
        MLX::Core.random_uniform(
          [texts.length, options[:image_size], options[:image_size], 3],
          0.0,
          255.0,
          MLX::Core.float32
        )
      end

      input_ids = MLX::Core.array(tokenizer.batch_encode(texts), MLX::Core.int32)
      pixel_values = processor.call(images)
      output = model.call(input_ids: input_ids, pixel_values: pixel_values, return_loss: true)
      MLX::Core.eval(output["logits_per_image"], output["loss"])

      puts "text_embeds_shape=#{output['text_embeds'].shape.inspect}"
      puts "image_embeds_shape=#{output['image_embeds'].shape.inspect}"
      puts "logits_per_image_shape=#{output['logits_per_image'].shape.inspect}"
      puts format("clip_loss=%.4f", output["loss"].item.to_f)

      logits = output["logits_per_image"].to_a
      logits.each_with_index do |row, i|
        best = row.each_with_index.max_by { |val, _j| val }
        puts format("image_%d best_text=%d score=%.4f text=%s", i, best[1], best[0], texts[best[1]])
      end
      output
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    texts: [],
    images_npz: nil,
    image_size: 224,
    patch_size: 16,
    embed_dim: 128,
    text_width: 256,
    vision_width: 256,
    context_length: 77
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby clip/clip.rb [options]"
    opts.on("--text TEXT", String, "Text prompt (repeatable)") { |v| options[:texts] << v }
    opts.on("--images-npz PATH", String, "Optional NPZ containing image batch [N,H,W,3]") { |v| options[:images_npz] = v }
    opts.on("--image-size N", Integer, "Image size") { |v| options[:image_size] = v }
    opts.on("--patch-size N", Integer, "Patch size") { |v| options[:patch_size] = v }
    opts.on("--embed-dim N", Integer, "Projection embedding dim") { |v| options[:embed_dim] = v }
    opts.on("--text-width N", Integer, "Text encoder width") { |v| options[:text_width] = v }
    opts.on("--vision-width N", Integer, "Vision encoder width") { |v| options[:vision_width] = v }
    opts.on("--context-length N", Integer, "Text context length") { |v| options[:context_length] = v }
  end
  parser.parse!

  ClipExample::Runner.run(options)
end
