# frozen_string_literal: true

require "open3"
require "optparse"

LLAMA_CONVERT = File.join(__dir__, "..", "llms", "llama", "convert.rb")

if $PROGRAM_NAME == __FILE__
  options = {
    torch_path: nil,
    mlx_path: "mlx_model",
    model_name: "llama",
    quantize: false,
    q_group_size: 64,
    q_bits: 4,
    dtype: "float16",
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby lora/convert.rb [options]"
    opts.on("--torch-path PATH", String, "Path to torch model files (required)") { |v| options[:torch_path] = v }
    opts.on("--mlx-path PATH", String, "Output MLX model directory") { |v| options[:mlx_path] = v }
    opts.on("--model-name NAME", String, "llama or tiny_llama") { |v| options[:model_name] = v }
    opts.on("-q", "--quantize", "Enable quantization") { options[:quantize] = true }
    opts.on("--q-group-size N", Integer, "Quantization group size") { |v| options[:q_group_size] = v }
    opts.on("--q-bits N", Integer, "Quantization bits") { |v| options[:q_bits] = v }
    opts.on("--dtype NAME", String, "Input/output dtype") { |v| options[:dtype] = v }
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  if options[:torch_path].nil? || options[:torch_path].empty?
    raise ArgumentError, "--torch-path is required"
  end

  args = [
    LLAMA_CONVERT,
    "--torch-path", options[:torch_path],
    "--mlx-path", options[:mlx_path],
    "--model-name", options[:model_name],
    "--q-group-size", options[:q_group_size].to_s,
    "--q-bits", options[:q_bits].to_s,
    "--dtype", options[:dtype],
    "--python-bin", options[:python_bin]
  ]
  args << "--quantize" if options[:quantize]

  stdout, stderr, status = Open3.capture3("ruby", *args)
  unless status.success?
    raise RuntimeError, "lora convert failed:\n#{stderr}"
  end

  puts stdout unless stdout.strip.empty?
  puts "[INFO] Saved converted base model to #{options[:mlx_path]}"
end
