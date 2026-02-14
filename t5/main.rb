# frozen_string_literal: true

require "optparse"
require "time"

require_relative "model"

if $PROGRAM_NAME == __FILE__
  options = {
    model: "t5-small",
    prompt: "translate English to German: That is good.",
    encode_only: false,
    max_tokens: 100,
    temp: 0.0,
    seed: 0,
    weights_path: nil,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby t5/main.rb [options]"
    opts.on("--model NAME", String, "Hugging Face T5 model name") { |v| options[:model] = v }
    opts.on("--prompt TEXT", String, "Prompt") { |v| options[:prompt] = v }
    opts.on("--encode-only", "Only run the encoder and print encoder output") { options[:encode_only] = true }
    opts.on("--max-tokens N", Integer, "Maximum generation tokens") { |v| options[:max_tokens] = v }
    opts.on("--temp N", Float, "Sampling temperature (0.0 = greedy)") { |v| options[:temp] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--weights-path PATH", String, "Path to converted .npz weights") { |v| options[:weights_path] = v }
    opts.on("--python-bin BIN", String, "Python binary for tokenizer/config bridges") { |v| options[:python_bin] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  model, tokenizer = T5Example.load_model(
    options[:model],
    weights_path: options[:weights_path],
    python_bin: options[:python_bin]
  )

  if options[:encode_only]
    puts "[INFO] Encoding with T5..."
    puts options[:prompt]
    prompt_ids = MLX::Core.array(tokenizer.encode(options[:prompt]), MLX::Core.int32)
    encoder_output = model.encode(MLX::Core.expand_dims(prompt_ids, 0))
    MLX::Core.eval(encoder_output)
    puts encoder_output
    exit(0)
  end

  puts "[INFO] Generating with T5..."
  puts "Input: #{options[:prompt]}"

  prompt_ids = MLX::Core.array(tokenizer.encode(options[:prompt]), MLX::Core.int32)
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  generated = 0

  T5Example.generate(
    prompt_ids,
    model,
    decoder_start_id: tokenizer.decoder_start_id,
    temp: options[:temp]
  ).each do |token|
    token_id = token.item.to_i
    break if token_id == tokenizer.eos_id

    print tokenizer.decode([token_id], with_sep: generated.positive?)
    generated += 1
    break if generated >= options[:max_tokens]
  end
  puts

  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
  tok_per_s = generated.zero? ? 0.0 : (generated / elapsed)
  puts format("Time: %.2f seconds, tokens/s: %.2f", elapsed, tok_per_s)
end
