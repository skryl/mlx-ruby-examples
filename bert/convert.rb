# frozen_string_literal: true

require "optparse"

require_relative "hf_bridge"

if $PROGRAM_NAME == __FILE__
  options = {
    bert_model: "bert-base-uncased",
    mlx_model: "weights/bert-base-uncased.npz",
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby bert/convert.rb [options]"
    opts.on("--bert-model NAME", String, "Hugging Face model name") { |v| options[:bert_model] = v }
    opts.on("--mlx-model PATH", String, "Output NPZ path") { |v| options[:mlx_model] = v }
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
  end
  parser.parse!

  bridge = BertExample::PythonBridge.new(python_bin: options[:python_bin])
  result = bridge.convert_weights(model_name: options[:bert_model], mlx_model: options[:mlx_model])

  puts "Saved weights: #{result.fetch('weights_path')}"
  puts "Saved config:  #{result.fetch('config_path')}"
end
