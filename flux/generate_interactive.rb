# frozen_string_literal: true

require "optparse"

require_relative "flux"
require_relative "txt2image"

module FluxExample
  module CLI
    module_function

    def print_help
      puts "The command list:"
      puts "- 'q' to exit"
      puts "- 's HxW' to change image size"
      puts "- 'n S' to change number of steps"
      puts "- 'h' to print this help"
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    quantize: false,
    model: "schnell",
    output: "flux/out.ppm"
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby flux/generate_interactive.rb [options]"
    opts.on("--quantize", "Quantize models") { options[:quantize] = true }
    opts.on("--model NAME", String, "schnell or dev") { |v| options[:model] = v }
    opts.on("--output PATH", String, "Output image path") { |v| options[:output] = v }
  end
  parser.parse!

  flux = FluxExample::FluxPipeline.new("flux-#{options[:model]}", t5_padding: true)

  if options[:quantize]
    begin
      MLX::NN.quantize(flux.flow)
      MLX::NN.quantize(flux.t5)
      MLX::NN.quantize(flux.clip)
    rescue StandardError
      # best effort
    end
  end

  puts "Loading models"
  flux.ensure_models_are_loaded

  puts "FLUX interactive session"
  FluxExample::CLI.print_help

  seed = 0
  size = [512, 512]
  latent_size = FluxExample::CLI.to_latent_size(size)
  steps = options[:model] == "dev" ? 50 : 4

  loop do
    print ">> "
    line = $stdin.gets
    break if line.nil?

    prompt = line.strip
    break if prompt == "q"

    if prompt == "h"
      FluxExample::CLI.print_help
      next
    end

    if prompt.start_with?("s ")
      size = prompt[2..].split("x").map(&:to_i)
      puts "Setting size to #{size.join('x')}"
      latent_size = FluxExample::CLI.to_latent_size(size)
      next
    end

    if prompt.start_with?("n ")
      steps = prompt[2..].to_i
      puts "Setting steps to #{steps}"
      next
    end

    seed += 1
    latents = flux.generate_latents(
      prompt,
      n_images: 1,
      num_steps: steps,
      latent_size: latent_size,
      guidance: 4.0,
      seed: seed
    )

    puts "Processing prompt"
    MLX::Core.eval(*latents.next)

    puts "Generating latents"
    x_t = nil
    latents.each do |xt|
      x_t = xt
      MLX::Core.eval(x_t)
    end

    puts "Generating image"
    img = flux.decode(x_t, latent_size)
    img = MLX::Core.squeeze(img, 0)
    MLX::Core.eval(img)
    FluxExample::CLI.write_ppm(options[:output], img)
    puts "Saved at #{options[:output]}"
    puts
  end
end
