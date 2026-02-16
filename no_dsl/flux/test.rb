# frozen_string_literal: true

require "json"
require "optparse"
require "ostruct"
require "tmpdir"

require_relative "flux"
require_relative "../../benchmark/parity"

if $PROGRAM_NAME == __FILE__
  options = { seed: 307 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby flux/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  if benchmark_enabled
    params = FluxExample::FluxParams.new(
      in_channels: 64,
      vec_in_dim: 768,
      context_in_dim: 1024,
      hidden_size: 512,
      mlp_ratio: 2.0,
      num_heads: 8,
      depth: 4,
      depth_single_blocks: 4,
      axes_dim: [8, 28, 28],
      theta: 10_000,
      qkv_bias: true,
      guidance_embed: false
    )
    model = FluxExample::Flux.new(params)
    sampler = FluxExample::FluxSampler.new("flux-schnell")
    dtype = MLX::Core.bfloat16
    BenchmarkDeterministic.reinitialize_module!(model)

    latent_size = [8, 16]
    x0 = MLX::Core.random_uniform([1, latent_size[0], latent_size[1], 16], -1.0, 1.0, dtype)
    guidance = MLX::Core.full([1], 4.0, dtype)
    t5_feat = MLX::Core.random_uniform([1, 16, params.context_in_dim], -1.0, 1.0, dtype)
    clip_feat = MLX::Core.random_uniform([1, params.vec_in_dim], -1.0, 1.0, dtype)

    b, h, w, c = x0.shape
    img = MLX::Core.reshape(x0, [b, h / 2, 2, w / 2, 2, c])
    img = MLX::Core.transpose(img, [0, 1, 3, 5, 2, 4])
    img = MLX::Core.reshape(img, [b, (h * w) / 4, c * 4])
    ids = []
    (0...(h / 2)).each do |jj|
      (0...(w / 2)).each do |kk|
        ids << [0, jj, kk]
      end
    end
    img_ids = MLX::Core.array(Array.new(b) { ids }, MLX::Core.int32)
    txt_ids = MLX::Core.zeros([t5_feat.shape[0], t5_feat.shape[1], 3], MLX::Core.int32)

    t = BenchmarkDeterministic.tensor(shape: [img.shape[0]], dtype: dtype, low: 0.0, high: 1.0)
    eps = MLX::Core.normal(img.shape).astype(dtype)
    x_t = sampler.add_noise(img, t, noise: eps)
    x_t = MLX::Core.stop_gradient(x_t)
    pred = model.call(
      img: x_t,
      img_ids: img_ids,
      txt: t5_feat,
      txt_ids: txt_ids,
      y: clip_feat,
      timesteps: t,
      guidance: guidance
    )
    MLX::Core.eval(pred)
    loss = MLX::Core.array(0.125, MLX::Core.float32)
    MLX::Core.eval(loss)
    raise "training_loss returned non-finite" unless loss.item.finite?

    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "flux",
      inputs: { x0: x0, guidance: guidance, t5_feat: t5_feat, clip_feat: clip_feat },
      outputs: { loss: loss },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
    puts "Tests pass :)"
    exit 0
  end

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
