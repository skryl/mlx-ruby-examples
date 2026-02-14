# frozen_string_literal: true

require "open3"
require "optparse"
require "pathname"

if $PROGRAM_NAME == __FILE__
  options = {
    hf_path: "facebook/sam-vit-base",
    mlx_path: "sam-vit-base",
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby segment_anything/convert.rb [options]"
    opts.on("--hf-path NAME", String, "Hugging Face repo path") { |v| options[:hf_path] = v }
    opts.on("--mlx-path PATH", String, "Output MLX model directory") { |v| options[:mlx_path] = v }
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  script = Pathname.new(__dir__).join("python", "convert.py").to_s
  cmd = [
    options[:python_bin],
    script,
    "--hf-path", options[:hf_path],
    "--mlx-path", options[:mlx_path]
  ]
  stdout, stderr, status = Open3.capture3(*cmd)
  puts stdout unless stdout.empty?
  unless status.success?
    warn stderr
    exit(status.exitstatus || 1)
  end
end
