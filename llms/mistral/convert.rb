# frozen_string_literal: true

require "open3"
require "optparse"

PY_SCRIPT_PATH = File.join(__dir__, "python", "convert.py")

if $PROGRAM_NAME == __FILE__
  options = {
    torch_path: "mistral-7B-v0.1",
    mlx_path: "mlx_model",
    quantize: false,
    q_group_size: 64,
    q_bits: 4,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/mistral/convert.rb [options]"
    opts.on("--torch-path PATH", String, "Path to PyTorch model directory") { |v| options[:torch_path] = v }
    opts.on("--mlx-path PATH", String, "Directory to save converted model") { |v| options[:mlx_path] = v }
    opts.on("-q", "--quantize", "Generate a quantized model") { options[:quantize] = true }
    opts.on("--q-group-size N", Integer, "Group size for quantization") { |v| options[:q_group_size] = v }
    opts.on("--q-bits N", Integer, "Bits per weight for quantization") { |v| options[:q_bits] = v }
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  args = [
    "--torch-path", options[:torch_path],
    "--mlx-path", options[:mlx_path],
    "--q-group-size", options[:q_group_size].to_s,
    "--q-bits", options[:q_bits].to_s
  ]
  args << "--quantize" if options[:quantize]

  stdout, stderr, status = Open3.capture3(options[:python_bin], PY_SCRIPT_PATH, *args)
  unless status.success?
    raise RuntimeError, "mistral convert failed:\n#{stderr}"
  end

  puts stdout unless stdout.empty?
  puts "[INFO] Saved converted model to #{options[:mlx_path]}"
end
