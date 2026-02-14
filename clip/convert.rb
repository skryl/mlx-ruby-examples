# frozen_string_literal: true

require "open3"
require "optparse"
require "pathname"

PY_SCRIPT = Pathname.new(__dir__).join("..", "mlx-examples", "clip", "convert.py").to_s

if $PROGRAM_NAME == __FILE__
  options = {
    hf_repo: "openai/clip-vit-base-patch32",
    mlx_path: "clip/mlx_model",
    dtype: "float32",
    force_download: false,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby clip/convert.rb [options]"
    opts.on("--hf-repo NAME", String, "Hugging Face repository") { |v| options[:hf_repo] = v }
    opts.on("--mlx-path PATH", String, "Output MLX model directory") { |v| options[:mlx_path] = v }
    opts.on("--dtype NAME", String, "Output dtype") { |v| options[:dtype] = v }
    opts.on("-f", "--force-download", "Force model redownload") { options[:force_download] = true }
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  args = [
    "--hf-repo", options[:hf_repo],
    "--mlx-path", options[:mlx_path],
    "--dtype", options[:dtype]
  ]
  args << "--force-download" if options[:force_download]

  stdout, stderr, status = Open3.capture3(options[:python_bin], PY_SCRIPT, *args)
  unless status.success?
    raise RuntimeError, "clip convert failed:\n#{stderr}"
  end
  puts stdout unless stdout.strip.empty?
end
