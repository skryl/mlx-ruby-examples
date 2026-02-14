# frozen_string_literal: true

require "optparse"

require_relative "flux"

module FluxExample
  module CLI
    module_function

    def to_latent_size(image_size)
      h, w = image_size
      h2 = ((h + 15) / 16) * 16
      w2 = ((w + 15) / 16) * 16

      if [h2, w2] != [h, w]
        puts "Warning: image dimensions must be divisible by 16px. Using #{h2}x#{w2}."
      end

      [h2 / 8, w2 / 8]
    end

    def quantization_predicate(_name, module_obj)
      module_obj.respond_to?(:to_quantized) && module_obj.respond_to?(:weight) && module_obj.weight.shape[1] % 512 == 0
    end

    def load_adapter(flux, adapter_file, fuse: false)
      return unless File.exist?(adapter_file)

      weights = MLX::Core.load(adapter_file)
      rank = 8
      num_blocks = -1
      flux.linear_to_lora_layers(rank: rank, num_blocks: num_blocks)
      flux.flow.load_weights(weights.to_a, strict: false)
      flux.fuse_lora_layers if fuse
    end

    def write_ppm(path, image)
      arr = image.to_a
      h = arr.length
      w = arr[0].length
      bytes = String.new(capacity: h * w * 3)
      arr.each do |row|
        row.each do |pix|
          r = [[(pix[0].to_f * 255.0).round, 0].max, 255].min
          g = [[(pix[1].to_f * 255.0).round, 0].max, 255].min
          b = [[(pix[2].to_f * 255.0).round, 0].max, 255].min
          bytes << r.chr << g.chr << b.chr
        end
      end
      File.open(path, "wb") do |f|
        f.write("P6\n#{w} #{h}\n255\n")
        f.write(bytes)
      end
    end

    def make_grid(x, n_rows)
      arr = x.to_a
      b = arr.length
      n_rows = [[n_rows.to_i, 1].max, b].min
      n_cols = (b.to_f / n_rows).ceil
      h = arr[0].length
      w = arr[0][0].length
      grid = Array.new(n_rows * h) { Array.new(n_cols * w) { [0.0, 0.0, 0.0] } }
      arr.each_with_index do |img, idx|
        row = idx / n_cols
        col = idx % n_cols
        h.times do |iy|
          w.times do |ix|
            grid[row * h + iy][col * w + ix] = img[iy][ix]
          end
        end
      end
      MLX::Core.array(grid, x.dtype)
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model: "schnell",
    n_images: 4,
    image_size: [512, 512],
    steps: nil,
    guidance: 4.0,
    n_rows: 1,
    decoding_batch_size: 1,
    output: "flux/out.ppm",
    save_raw: false,
    seed: nil,
    verbose: false,
    adapter: nil,
    fuse_adapter: false,
    t5_padding: true,
    quantize: false,
    preload_models: false,
    force_shard: false
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby flux/txt2image.rb [options] PROMPT"
    opts.on("--model NAME", String, "schnell or dev") { |v| options[:model] = v }
    opts.on("--n-images N", Integer, "Number of images") { |v| options[:n_images] = v }
    opts.on("--image-size HxW", String, "Output size, e.g. 512x512") do |v|
      options[:image_size] = v.split("x").map(&:to_i)
    end
    opts.on("--steps N", Integer, "Denoising steps") { |v| options[:steps] = v }
    opts.on("--guidance N", Float, "Guidance factor") { |v| options[:guidance] = v }
    opts.on("--n-rows N", Integer, "Rows in output grid") { |v| options[:n_rows] = v }
    opts.on("--decoding-batch-size N", Integer, "Decode batch size") { |v| options[:decoding_batch_size] = v }
    opts.on("--output PATH", String, "Output path") { |v| options[:output] = v }
    opts.on("--save-raw", "Save each generated image separately") { options[:save_raw] = true }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--verbose", "Print generation diagnostics") { options[:verbose] = true }
    opts.on("--adapter PATH", String, "Optional adapter weights") { |v| options[:adapter] = v }
    opts.on("--fuse-adapter", "Fuse adapter into base model") { options[:fuse_adapter] = true }
    opts.on("--no-t5-padding", "Disable T5 tokenizer padding") { options[:t5_padding] = false }
    opts.on("--quantize", "Apply quantization to models") { options[:quantize] = true }
    opts.on("--preload-models", "Eagerly load model parameters") { options[:preload_models] = true }
    opts.on("--force-shard", "Reserved distributed option (no-op in Ruby port)") { options[:force_shard] = true }
  end
  parser.parse!

  prompt = ARGV.join(" ").strip
  raise OptionParser::MissingArgument, "PROMPT" if prompt.empty?

  flux = FluxExample::FluxPipeline.new("flux-#{options[:model]}", t5_padding: options[:t5_padding])
  options[:steps] ||= (options[:model] == "dev" ? 50 : 2)

  if !options[:adapter].nil?
    FluxExample::CLI.load_adapter(flux, options[:adapter], fuse: options[:fuse_adapter])
  end

  if options[:quantize]
    begin
      MLX::NN.quantize(flux.flow)
      MLX::NN.quantize(flux.t5)
      MLX::NN.quantize(flux.clip)
    rescue StandardError
      # Keep running even if quantization is unsupported in this environment.
    end
  end

  flux.ensure_models_are_loaded if options[:preload_models]

  latent_size = FluxExample::CLI.to_latent_size(options[:image_size])
  latents = flux.generate_latents(
    prompt,
    n_images: options[:n_images],
    num_steps: options[:steps],
    latent_size: latent_size,
    guidance: options[:guidance],
    seed: options[:seed]
  )

  conditioning = latents.next
  MLX::Core.eval(*conditioning)

  flux.reload_text_encoders

  x_t = nil
  latents.each do |step_latent|
    x_t = step_latent
    MLX::Core.eval(x_t)
  end

  decoded = []
  step = [options[:decoding_batch_size], 1].max
  (0...options[:n_images]).step(step) do |i|
    x = MLX::Core.slice(x_t, [i, 0, 0], [[i + step, options[:n_images]].min, x_t.shape[1], x_t.shape[2]])
    img = flux.decode(x, latent_size)
    MLX::Core.eval(img)
    decoded << img
  end
  decoded = MLX::Core.concatenate(decoded, 0)
  MLX::Core.eval(decoded)

  if options[:save_raw]
    stem = options[:output].sub(/\.[^.]+$/, "")
    ext = File.extname(options[:output])
    ext = ".ppm" if ext.empty?
    decoded.shape[0].times do |i|
      img = MLX::Core.squeeze(
        MLX::Core.slice(decoded, [i, 0, 0, 0], [i + 1, decoded.shape[1], decoded.shape[2], decoded.shape[3]]),
        0
      )
      FluxExample::CLI.write_ppm("#{stem}.#{i}#{ext}", img)
    end
  else
    grid = FluxExample::CLI.make_grid(decoded, options[:n_rows])
    MLX::Core.eval(grid)
    FluxExample::CLI.write_ppm(options[:output], grid)
  end

  puts "saved=#{options[:output]}"
end
