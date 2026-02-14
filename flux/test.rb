# frozen_string_literal: true

require "json"
require "optparse"
require "ostruct"
require "tmpdir"

require_relative "flux"

if $PROGRAM_NAME == __FILE__
  options = { seed: 307 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby flux/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  flux = FluxExample::FluxPipeline.new("flux-schnell", hf_download: false)

  t5_tokens, clip_tokens = flux.tokenize("A photo of an astronaut riding a horse on Mars")
  MLX::Core.eval(t5_tokens, clip_tokens)
  raise "T5 token batch mismatch" unless t5_tokens.shape[0] == 1
  raise "CLIP token batch mismatch" unless clip_tokens.shape[0] == 1

  latent_size = [8, 16]
  latents = flux.generate_latents(
    "A photo of an astronaut riding a horse on Mars",
    n_images: 2,
    num_steps: 3,
    latent_size: latent_size,
    guidance: 4.0,
    seed: options[:seed]
  )

  conditioning = latents.next
  x_t0, x_ids, txt, txt_ids, vec = conditioning
  MLX::Core.eval(x_t0, x_ids, txt, txt_ids, vec)
  raise "Initial latent shape mismatch: #{x_t0.shape.inspect}" unless x_t0.shape[0] == 2
  raise "Image id shape mismatch" unless x_ids.shape[0] == 2 && x_ids.shape[2] == 3
  raise "T5 conditioning batch mismatch" unless txt.shape[0] == 2
  raise "T5 position ids shape mismatch" unless txt_ids.shape[0] == 2 && txt_ids.shape[2] == 3
  raise "CLIP conditioning batch mismatch" unless vec.shape[0] == 2

  x_t = nil
  latents.each do |xt|
    x_t = xt
    MLX::Core.eval(x_t)
  end
  raise "Missing denoised latent output" if x_t.nil?

  decoded = flux.decode(x_t, latent_size)
  MLX::Core.eval(decoded)
  unless decoded.shape == [2, latent_size[0] * 8, latent_size[1] * 8, 3]
    raise "Decoded image shape mismatch: #{decoded.shape.inspect}"
  end

  generated = flux.generate_images(
    "A tiny robot in watercolor",
    n_images: 2,
    num_steps: 2,
    latent_size: latent_size,
    seed: options[:seed],
    progress: false
  )
  MLX::Core.eval(generated)
  raise "generate_images output shape mismatch" unless generated.shape == [2, latent_size[0] * 8, latent_size[1] * 8, 3]

  x0 = MLX::Core.random_uniform([1, latent_size[0], latent_size[1], 16], -1.0, 1.0, flux.dtype)
  guidance = MLX::Core.full([1], 4.0, flux.dtype)
  t5_feat = flux.t5.call(flux.t5_tokenizer.encode(["test prompt"], pad: true))
  clip_feat = flux.clip.call(flux.clip_tokenizer.encode(["test prompt"])).pooled_output
  loss = flux.training_loss(x0, t5_feat, clip_feat, guidance)
  MLX::Core.eval(loss)
  raise "training_loss returned non-finite" unless loss.item.finite?

  flux.linear_to_lora_layers(rank: 4, num_blocks: 2)
  flux.fuse_lora_layers

  Dir.mktmpdir("flux-test-") do |dir|
    train_file = File.join(dir, "train.jsonl")
    File.binwrite(train_file, JSON.generate({ "image" => "missing.ppm", "prompt" => "a sample" }) + "\n")

    dataset = FluxExample.load_dataset(dir)
    trainer = FluxExample::Trainer.new(
      flux,
      dataset,
      OpenStruct.new(
        resolution: [64, 64],
        num_augmentations: 2
      )
    )
    trainer.encode_dataset
    iter = trainer.iterate(1)
    bx, bt5, bclip = iter.next
    MLX::Core.eval(bx, bt5, bclip)
    raise "Trainer batch x shape mismatch" unless bx.shape[0] == 1
    raise "Trainer batch t5 shape mismatch" unless bt5.shape[0] == 1
    raise "Trainer batch clip shape mismatch" unless bclip.shape[0] == 1
  end

  puts "Tests pass :)"
end
