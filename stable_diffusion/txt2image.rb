# frozen_string_literal: true

require "optparse"

require_relative "stable_diffusion"

module StableDiffusionExample
  module CLI
    module_function

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
    model: "sdxl",
    n_images: 4,
    steps: nil,
    cfg: nil,
    negative_prompt: "",
    n_rows: 1,
    output: "stable_diffusion/out.ppm",
    seed: nil,
    float16: false
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby stable_diffusion/txt2image.rb [options] PROMPT"
    opts.on("--model NAME", String, "sd or sdxl") { |v| options[:model] = v }
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

  prompt = ARGV.join(" ").strip
  raise OptionParser::MissingArgument, "PROMPT" if prompt.empty?

  sd = if options[:model] == "sdxl"
    options[:cfg] = 0.0 if options[:cfg].nil?
    options[:steps] = 2 if options[:steps].nil?
    StableDiffusionExample::StableDiffusionXL.new(float16: options[:float16])
  else
    options[:cfg] = 7.5 if options[:cfg].nil?
    options[:steps] = 20 if options[:steps].nil?
    StableDiffusionExample::StableDiffusion.new(float16: options[:float16])
  end

  x_t = nil
  sd.generate_latents(
    prompt,
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
