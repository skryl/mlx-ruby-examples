# frozen_string_literal: true

require "open3"
require "optparse"

PY_SCRIPT_PATH = File.join(__dir__, "python", "convert.py")

if $PROGRAM_NAME == __FILE__
  options = {
    model: "t5-small",
    dtype: "float32",
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/speculative_decoding/convert.rb [options]"
    opts.on("--model NAME", String, "Name of the T5 model") { |v| options[:model] = v }
    opts.on("--dtype NAME", String, "float16 or float32") { |v| options[:dtype] = v }
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  unless %w[float16 float32].include?(options[:dtype])
    raise ArgumentError, "--dtype must be one of float16, float32"
  end

  args = [
    "--model", options[:model],
    "--dtype", options[:dtype]
  ]

  stdout, stderr, status = Open3.capture3(options[:python_bin], PY_SCRIPT_PATH, *args)
  unless status.success?
    raise RuntimeError, "speculative convert failed:\n#{stderr}"
  end

  puts stdout unless stdout.empty?
end
