# frozen_string_literal: true

require "json"
require "optparse"
require "pathname"

require_relative "flux"
require_relative "txt2image"

module FluxExample
  module Dreambooth
    module_function

    def generate_progress_images(iteration, flux, options)
      out_dir = Pathname.new(options[:output_dir])
      out_dir.mkpath
      out_file = out_dir.join(format("%07d_progress.ppm", iteration))
      puts "Generating #{out_file}"

      n_rows = 2
      n_images = 4
      x = flux.generate_images(
        options[:progress_prompt],
        n_images: n_images,
        num_steps: options[:progress_steps],
        latent_size: FluxExample::CLI.to_latent_size(options[:resolution]),
        progress: false
      )
      grid = FluxExample::CLI.make_grid(x, n_rows)
      MLX::Core.eval(grid)
      FluxExample::CLI.write_ppm(out_file.to_s, grid)
    end

    def save_adapters(adapter_name, flux, options)
      out_dir = Pathname.new(options[:output_dir])
      out_dir.mkpath
      out_file = out_dir.join(adapter_name)
      puts "Saving #{out_file}"
      flux.flow.save_weights(out_file.to_s)

      meta_path = out_dir.join("#{adapter_name}.json")
      meta = {
        "lora_rank" => options[:lora_rank],
        "lora_blocks" => options[:lora_blocks]
      }
      File.binwrite(meta_path.to_s, JSON.pretty_generate(meta))
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model: "dev",
    guidance: 4.0,
    iterations: 200,
    batch_size: 1,
    resolution: [512, 512],
    num_augmentations: 2,
    progress_prompt: nil,
    progress_steps: 8,
    progress_every: 50,
    checkpoint_every: 50,
    lora_blocks: -1,
    lora_rank: 8,
    warmup_steps: 50,
    learning_rate: 1e-4,
    grad_accumulate: 2,
    output_dir: "flux/mlx_output"
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby flux/dreambooth.rb [options] DATASET"
    opts.on("--model NAME", String, "dev or schnell") { |v| options[:model] = v }
    opts.on("--guidance N", Float, "Guidance factor") { |v| options[:guidance] = v }
    opts.on("--iterations N", Integer, "Train iterations") { |v| options[:iterations] = v }
    opts.on("--batch-size N", Integer, "Batch size") { |v| options[:batch_size] = v }
    opts.on("--resolution HxW", String, "Training resolution") { |v| options[:resolution] = v.split("x").map(&:to_i) }
    opts.on("--num-augmentations N", Integer, "Image augmentations per sample") { |v| options[:num_augmentations] = v }
    opts.on("--progress-prompt TEXT", String, "Prompt for periodic generations") { |v| options[:progress_prompt] = v }
    opts.on("--progress-steps N", Integer, "Steps for progress generations") { |v| options[:progress_steps] = v }
    opts.on("--progress-every N", Integer, "Progress generation interval") { |v| options[:progress_every] = v }
    opts.on("--checkpoint-every N", Integer, "Checkpoint interval") { |v| options[:checkpoint_every] = v }
    opts.on("--lora-blocks N", Integer, "Number of trainable blocks") { |v| options[:lora_blocks] = v }
    opts.on("--lora-rank N", Integer, "LoRA rank") { |v| options[:lora_rank] = v }
    opts.on("--warmup-steps N", Integer, "LR warmup steps") { |v| options[:warmup_steps] = v }
    opts.on("--learning-rate N", Float, "Learning rate") { |v| options[:learning_rate] = v }
    opts.on("--grad-accumulate N", Integer, "Gradient accumulation") { |v| options[:grad_accumulate] = v }
    opts.on("--output-dir PATH", String, "Output folder") { |v| options[:output_dir] = v }
  end
  parser.parse!

  dataset_name = ARGV.shift
  raise OptionParser::MissingArgument, "DATASET" if dataset_name.nil? || dataset_name.empty?
  if options[:progress_prompt].nil? || options[:progress_prompt].empty?
    raise OptionParser::MissingArgument, "--progress-prompt"
  end

  output_path = Pathname.new(options[:output_dir])
  output_path.mkpath
  FluxExample.save_config(options, output_path.join("adapter_config.json"))

  MLX::Core.random_seed(0x0F0F0F0F)
  flux = FluxExample::FluxPipeline.new("flux-#{options[:model]}")
  flux.linear_to_lora_layers(rank: options[:lora_rank], num_blocks: options[:lora_blocks])

  total_params = MLX::Utils.tree_flatten(flux.flow.trainable_parameters).sum { |_k, v| v.size }
  puts format("Training %.3fM parameters", total_params / 1024.0 / 1024.0)

  optimizer = MLX::Optimizers::Adam.new(learning_rate: options[:learning_rate])

  dataset = FluxExample.load_dataset(dataset_name)
  trainer = FluxExample::Trainer.new(flux, dataset, OpenStruct.new(options))
  trainer.encode_dataset

  guidance = MLX::Core.full([options[:batch_size]], options[:guidance], flux.dtype)

  FluxExample::Dreambooth.generate_progress_images(0, flux, options)

  grads_acc = nil
  losses = []
  step_fn = MLX::NN.value_and_grad(
    flux.flow,
    lambda do |x, t5_feat, clip_feat, guidance_vec|
      flux.training_loss(x, t5_feat, clip_feat, guidance_vec)
    end
  )

  iter = trainer.iterate(options[:batch_size])
  MLX::DSL::Data.from(0...options[:iterations]).each do |i|
    x, t5_feat, clip_feat = iter.next
    loss, grads = step_fn.call(x, t5_feat, clip_feat, guidance)

    if grads_acc.nil?
      grads_acc = grads
    else
      grads_acc = MLX::Utils.tree_map(lambda { |a, b| MLX::Core.add(a, b) }, grads_acc, grads)
    end

    if ((i + 1) % options[:grad_accumulate]).zero?
      scale = 1.0 / options[:grad_accumulate]
      grads_acc = MLX::Utils.tree_map(lambda { |g| MLX::Core.multiply(scale, g) }, grads_acc)
      optimizer.update(flux.flow, grads_acc)
      grads_acc = nil
      MLX::Core.eval(flux.flow.parameters, optimizer.state)
    end

    MLX::Core.eval(loss)
    losses << loss.item

    if ((i + 1) % 10).zero?
      avg = losses.sum / losses.length
      puts format("Iter: %d Loss: %.4f", i + 1, avg)
      losses.clear
    end

    if ((i + 1) % options[:progress_every]).zero?
      FluxExample::Dreambooth.generate_progress_images(i + 1, flux, options)
    end

    if ((i + 1) % options[:checkpoint_every]).zero?
      FluxExample::Dreambooth.save_adapters(format("%07d_adapters.npz", i + 1), flux, options)
    end
  end

  FluxExample::Dreambooth.save_adapters("final_adapters.npz", flux, options)
  puts "Training successful."
end
