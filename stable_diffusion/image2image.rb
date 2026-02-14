# frozen_string_literal: true

require "optparse"

require_relative "stable_diffusion"
require_relative "txt2image"

module StableDiffusionExample
  module CLI
    module_function

    def read_ppm(path)
      data = File.binread(path)
      idx = 0

      read_token = lambda do
        idx += 1 while idx < data.length && data.getbyte(idx) <= 32
        if idx < data.length && data.getbyte(idx) == 35
          idx += 1 while idx < data.length && data.getbyte(idx) != 10
          idx += 1
          idx += 1 while idx < data.length && data.getbyte(idx) <= 32
        end
        start = idx
        idx += 1 while idx < data.length && data.getbyte(idx) > 32
        data[start...idx]
      end

      magic = read_token.call
      raise "Only P6 PPM is supported" unless magic == "P6"

      w = read_token.call.to_i
      h = read_token.call.to_i
      maxv = read_token.call.to_i
      raise "Invalid PPM max value #{maxv}" unless maxv.positive?

      idx += 1 while idx < data.length && data.getbyte(idx) <= 32
      payload = data.byteslice(idx, w * h * 3)
      raise "PPM payload is truncated" if payload.nil? || payload.bytesize < w * h * 3

      arr = Array.new(h) { Array.new(w) { [0.0, 0.0, 0.0] } }
      p = 0
      h.times do |y|
        w.times do |x|
          r = payload.getbyte(p)
          g = payload.getbyte(p + 1)
          b = payload.getbyte(p + 2)
          p += 3
          arr[y][x] = [r / 255.0, g / 255.0, b / 255.0]
        end
      end
      MLX::Core.array(arr, MLX::Core.float32)
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model: "sdxl",
    strength: 0.9,
    n_images: 4,
    steps: nil,
    cfg: nil,
    negative_prompt: "",
    n_rows: 1,
    output: "stable_diffusion/im2im_out.ppm",
    seed: nil,
    float16: false
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby stable_diffusion/image2image.rb [options] IMAGE.ppm PROMPT"
    opts.on("--model NAME", String, "sd or sdxl") { |v| options[:model] = v }
    opts.on("--strength N", Float, "Image conditioning strength") { |v| options[:strength] = v }
    opts.on("--n-images N", Integer, "Number of images") { |v| options[:n_images] = v }
    opts.on("--steps N", Integer, "Denoising steps") { |v| options[:steps] = v }
    opts.on("--cfg N", Float, "Classifier free guidance weight") { |v| options[:cfg] = v }
    opts.on("--negative-prompt TEXT", String, "Negative prompt") { |v| options[:negative_prompt] = v }
    opts.on("--n-rows N", Integer, "Rows in output grid") { |v| options[:n_rows] = v }
    opts.on("--output PATH", String, "Output PPM image path") { |v| options[:output] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--float16", "Use float16") { options[:float16] = true }
  end
  parser.parse!

  raise OptionParser::MissingArgument, "IMAGE.ppm" if ARGV.empty?
  image_path = ARGV.shift
  prompt = ARGV.join(" ").strip
  raise OptionParser::MissingArgument, "PROMPT" if prompt.empty?

  image = StableDiffusionExample::CLI.read_ppm(image_path)
  h = image.shape[0] - (image.shape[0] % 64)
  w = image.shape[1] - (image.shape[1] % 64)
  image = MLX::Core.slice(image, [0, 0, 0], [h, w, image.shape[2]])
  image = MLX::Core.subtract(MLX::Core.multiply(image, 2.0), 1.0)

  sd = if options[:model] == "sdxl"
    options[:cfg] = 0.0 if options[:cfg].nil?
    options[:steps] = 2 if options[:steps].nil?
    StableDiffusionExample::StableDiffusionXL.new(float16: options[:float16])
  else
    options[:cfg] = 7.5 if options[:cfg].nil?
    options[:steps] = 20 if options[:steps].nil?
    StableDiffusionExample::StableDiffusion.new(float16: options[:float16])
  end

  if (options[:steps] * options[:strength]).to_i < 1
    options[:steps] = (1.0 / options[:strength]).ceil
  end

  x_t = nil
  sd.generate_latents_from_image(
    image,
    prompt,
    strength: options[:strength],
    n_images: options[:n_images],
    cfg_weight: options[:cfg],
    num_steps: options[:steps],
    negative_text: options[:negative_prompt],
    seed: options[:seed]
  ).each do |latent|
    x_t = latent
    MLX::Core.eval(x_t)
  end

  decoded = sd.decode(x_t)
  MLX::Core.eval(decoded)
  grid = StableDiffusionExample::CLI.make_grid(decoded, options[:n_rows])
  MLX::Core.eval(grid)

  StableDiffusionExample::CLI.write_ppm(options[:output], grid)
  puts "saved=#{options[:output]}"
end
