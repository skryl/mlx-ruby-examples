# frozen_string_literal: true

require "optparse"
require "time"

require_relative "decoder"
require_relative "model"

module SpeculativeDecodingExample
  module_function

  def load_model(model_name, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    config = SpeculativeDecodingExample.load_t5_config(model_name, python_bin: python_bin)
    model = SpeculativeDecodingExample::Model.new(config)
    weights = MLX::Core.load("#{model_name.tr('/', '-')}.npz")
    model.update(MLX::Utils.tree_unflatten(weights.to_a))
    MLX::Core.eval(model.parameters)
    model
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    num_draft: 5,
    model_name: "t5-small",
    draft_model_name: "t5-small",
    seed: 0,
    max_tokens: 100,
    prompt: "translate English to French: Let's go to the store and buy some groceries including eggs, avocadoes, and bread.",
    delta: 0.1,
    regular_decode: false,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/speculative_decoding/main.rb [options]"
    opts.on("--num-draft N", Integer, "Number of draft tokens per decoding step") { |v| options[:num_draft] = v }
    opts.on("--model-name NAME", String, "Main model name") { |v| options[:model_name] = v }
    opts.on("--draft-model-name NAME", String, "Draft model name") { |v| options[:draft_model_name] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--max-tokens N", Integer, "Maximum number of generated tokens") { |v| options[:max_tokens] = v }
    opts.on("--prompt TEXT", String, "Prompt text") { |v| options[:prompt] = v }
    opts.on("--delta N", Float, "Lenience for accepting proposal tokens") { |v| options[:delta] = v }
    opts.on("--regular-decode", "Use regular decoding instead of speculative decoding") do
      options[:regular_decode] = true
    end
    opts.on("--python-bin BIN", String, "Python binary for HF bridges") { |v| options[:python_bin] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  spec_decoder = SpeculativeDecodingExample::SpeculativeDecoder.new(
    model: SpeculativeDecodingExample.load_model(options[:model_name], python_bin: options[:python_bin]),
    draft_model: SpeculativeDecodingExample.load_model(options[:draft_model_name], python_bin: options[:python_bin]),
    tokenizer: options[:model_name],
    delta: options[:delta],
    num_draft: options[:num_draft],
    python_bin: options[:python_bin]
  )

  start = Time.now
  puts options[:prompt]
  if options[:regular_decode]
    spec_decoder.generate(options[:prompt], max_tokens: options[:max_tokens])
  else
    stats = spec_decoder.speculative_decode(options[:prompt], max_tokens: options[:max_tokens])
    puts "=" * 10
    puts "Accepted #{stats.fetch('n_accepted')} / #{stats.fetch('n_draft')}."
    puts "Decoding steps #{stats.fetch('n_steps')}."
  end
  puts "=" * 10
  puts format("Full generation time %.3f", Time.now - start)
end
