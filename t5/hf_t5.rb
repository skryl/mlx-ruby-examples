# frozen_string_literal: true

require "json"
require "open3"
require "optparse"

PY_SCRIPT = File.join(__dir__, "python", "hf_t5_bridge.py")

if $PROGRAM_NAME == __FILE__
  options = {
    encode_only: false,
    model: "t5-small",
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby t5/hf_t5.rb [options]"
    opts.on("--encode-only", "Only run the encoder and print embeddings") { options[:encode_only] = true }
    opts.on("--model NAME", String, "Hugging Face T5 model") { |v| options[:model] = v }
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  args = ["--model", options[:model]]
  args << "--encode-only" if options[:encode_only]
  stdout, stderr, status = Open3.capture3(options[:python_bin], PY_SCRIPT, *args)
  unless status.success?
    raise RuntimeError, "hf_t5 bridge failed:\n#{stderr}"
  end

  payload = JSON.parse(stdout)
  if options[:encode_only]
    batch = payload.fetch("batch")
    embedding = payload.fetch("embedding")
    puts "\nHF T5:"
    batch.each_with_index do |input_str, i|
      puts "Input: #{input_str}"
      puts embedding[i].inspect
      puts
    end
  else
    puts payload.fetch("output")
  end
end
