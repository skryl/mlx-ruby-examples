# frozen_string_literal: true

require "open3"
require "optparse"
require "pathname"

PY_SCRIPT = Pathname.new(__dir__).join("..", "mlx-examples", "clip", "hf_preproc.py").to_s

if $PROGRAM_NAME == __FILE__
  options = {
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby clip/hf_preproc.rb [options]"
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  stdout, stderr, status = Open3.capture3(options[:python_bin], PY_SCRIPT)
  unless status.success?
    raise RuntimeError, "clip hf_preproc failed:\n#{stderr}"
  end

  puts stdout unless stdout.strip.empty?
end
