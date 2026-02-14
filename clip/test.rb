# frozen_string_literal: true

require "optparse"
require "tmpdir"

require_relative "clip"
require_relative "linear_probe"

if $PROGRAM_NAME == __FILE__
  options = { seed: 67 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby clip/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  tokenizer = ClipExample::SimpleTokenizer.new(max_length: 16)
  encoded = tokenizer.encode("cat")
  raise "Tokenizer length mismatch" unless encoded.length == 16
  raise "Tokenizer bos/eos mismatch" unless encoded[0] == tokenizer.bos_id

  processor = ClipExample::ImageProcessor.new(image_size: 32)
  raw = MLX::Core.random_uniform([2, 40, 40, 3], 0.0, 255.0, MLX::Core.float32)
  proc = processor.call(raw)
  MLX::Core.eval(proc)
  raise "Image processor output shape mismatch: #{proc.shape.inspect}" unless proc.shape == [2, 32, 32, 3]

  model = ClipExample::CLIPModel.new(
    vocab_size: tokenizer.vocab_size,
    text_width: 64,
    vision_width: 64,
    embed_dim: 32,
    image_size: 32,
    patch_size: 8,
    max_length: 16
  )
  input_ids = MLX::Core.array(tokenizer.batch_encode(["a cat", "a dog"]), MLX::Core.int32)
  output = model.call(input_ids: input_ids, pixel_values: proc, return_loss: true)
  MLX::Core.eval(output["logits_per_image"], output["loss"])
  raise "Text embeds shape mismatch" unless output["text_embeds"].shape == [2, 32]
  raise "Image embeds shape mismatch" unless output["image_embeds"].shape == [2, 32]
  raise "Logits shape mismatch" unless output["logits_per_image"].shape == [2, 2]
  raise "Loss not finite" unless output["loss"].item.finite?

  before = MLX::Core.array(model.vision_projection.weight.to_a, model.vision_projection.weight.dtype)
  optimizer = MLX::Optimizers::Adam.new(learning_rate: 1e-3)
  step = MLX::NN.value_and_grad(
    model,
    lambda do |ids, imgs|
      out = model.call(input_ids: ids, pixel_values: imgs, return_loss: true)
      out.fetch("loss")
    end
  )
  loss, grads = step.call(input_ids, proc)
  optimizer.update(model, grads)
  MLX::Core.eval(loss, model.parameters, optimizer.state)
  delta = MLX::Core.sum(MLX::Core.abs(MLX::Core.subtract(model.vision_projection.weight, before)))
  MLX::Core.eval(delta)
  raise "Optimizer step did not update vision projection" if delta.item <= 0.0

  ClipExample::Runner.run(
    texts: ["a photo of a cat", "a photo of a dog"],
    images_npz: nil,
    image_size: 32,
    patch_size: 8,
    embed_dim: 32,
    text_width: 64,
    vision_width: 64,
    context_length: 16
  )

  Dir.mktmpdir("clip-test-") do |dir|
    system(
      "ruby",
      File.join(__dir__, "linear_probe.rb"),
      "--samples", "64",
      "--classes", "4",
      "--image-size", "32",
      "--batch-size", "16",
      "--epochs", "2",
      "--lr", "0.01",
      "--seed", options[:seed].to_s
    ) or raise "linear_probe smoke run failed"
  end

  puts "Tests pass :)"
end
