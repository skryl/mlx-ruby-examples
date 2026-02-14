# frozen_string_literal: true

require "open3"
require "optparse"

PY_SCRIPT = File.join(__dir__, "python", "convert.py")

if $PROGRAM_NAME == __FILE__
  options = {
    model: "t5-small",
    dtype: "float32",
    output: nil,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby t5/convert.rb [options]"
    opts.on("--model NAME", String, "Hugging Face T5 model name") { |v| options[:model] = v }
    opts.on("--dtype NAME", String, "float16 or float32") { |v| options[:dtype] = v }
    opts.on("--output PATH", String, "Output .npz path") { |v| options[:output] = v }
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  unless %w[float16 float32].include?(options[:dtype])
    raise ArgumentError, "--dtype must be one of float16, float32"
  end

  args = ["--model", options[:model], "--dtype", options[:dtype]]
  args += ["--output", options[:output]] unless options[:output].nil? || options[:output].empty?

  stdout, stderr, status = Open3.capture3(options[:python_bin], PY_SCRIPT, *args)
  unless status.success?
    raise RuntimeError, "t5 convert failed:\n#{stderr}"
  end

  puts stdout unless stdout.strip.empty?
end
