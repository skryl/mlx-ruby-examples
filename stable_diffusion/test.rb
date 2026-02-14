# frozen_string_literal: true

require "optparse"

require_relative "stable_diffusion"

if $PROGRAM_NAME == __FILE__
  options = { seed: 211 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby stable_diffusion/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  sd = StableDiffusionExample::StableDiffusion.new(float16: false)
  latent = nil
  sd.generate_latents(
    "a red cube",
    n_images: 2,
    num_steps: 3,
    cfg_weight: 2.0,
    negative_text: "blurry",
    latent_size: [8, 8],
    seed: options[:seed]
  ).each do |x_t|
    latent = x_t
  end
  MLX::Core.eval(latent)
  raise "StableDiffusion latent shape mismatch #{latent.shape.inspect}" unless latent.shape == [2, 8, 8, sd.autoencoder.latent_channels]

  decoded = sd.decode(latent)
  MLX::Core.eval(decoded)
  raise "StableDiffusion decode shape mismatch #{decoded.shape.inspect}" unless decoded.shape == [2, 64, 64, 3]

  image = MLX::Core.random_uniform([64, 64, 3], -1.0, 1.0, MLX::Core.float32)
  latent_img = nil
  sd.generate_latents_from_image(
    image,
    "a blue cube",
    n_images: 2,
    strength: 0.5,
    num_steps: 4,
    cfg_weight: 1.0,
    seed: options[:seed]
  ).each do |x_t|
    latent_img = x_t
  end
  MLX::Core.eval(latent_img)
  raise "image2image latent shape mismatch #{latent_img.shape.inspect}" unless latent_img.shape == [2, 8, 8, sd.autoencoder.latent_channels]

  steps = sd.sampler.timesteps(4, start_time: sd.sampler.max_time, dtype: MLX::Core.float32).to_a
  raise "Sampler timestep count mismatch" unless steps.length == 4

  sdxl = StableDiffusionExample::StableDiffusionXL.new(float16: false)
  latent_xl = nil
  sdxl.generate_latents(
    "a green cube",
    n_images: 1,
    num_steps: 2,
    cfg_weight: 0.0,
    latent_size: [8, 8],
    seed: options[:seed]
  ).each do |x_t|
    latent_xl = x_t
  end
  MLX::Core.eval(latent_xl)
  raise "StableDiffusionXL latent shape mismatch #{latent_xl.shape.inspect}" unless latent_xl.shape == [1, 8, 8, sdxl.autoencoder.latent_channels]

  puts "Tests pass :)"
end
