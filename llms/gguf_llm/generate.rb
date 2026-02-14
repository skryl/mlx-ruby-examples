# frozen_string_literal: true

require "optparse"
require "time"

require_relative "models"

module GGUFLLM
  module_function

  def run_generate(model:, tokenizer:, prompt:, max_tokens:, temp:)
    prompt_tokens = tokenizer.encode(prompt)
    start = Time.now
    tokens = []
    skip = 0
    prompt_time = nil

    GGUFLLM.generate(prompt_tokens, model, temp: temp).each_with_index do |token, n|
      break if token.item.to_i == tokenizer.eos_token_id

      if n.zero?
        prompt_time = Time.now - start
        start = Time.now
      end

      tokens << token.item.to_i
      s = tokenizer.decode(tokens)
      chunk = s[skip..]
      print(chunk, end: "", flush: true) unless chunk.nil?
      skip = s.length

      break if n + 1 >= max_tokens
    end

    tail = tokenizer.decode(tokens)
    chunk = tail[skip..]
    print(chunk, flush: true) unless chunk.nil?

    gen_time = Time.now - start
    puts "=" * 10
    if tokens.empty?
      puts "No tokens generated for this prompt"
      return
    end
    prompt_tps = prompt_tokens.size / [prompt_time, 1e-9].max
    gen_tps = (tokens.length - 1) / [gen_time, 1e-9].max
    puts format("Prompt: %.3f tokens-per-sec", prompt_tps)
    puts format("Generation: %.3f tokens-per-sec", gen_tps)
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    gguf: nil,
    repo: nil,
    prompt: "In the beginning the Universe was created.",
    max_tokens: 100,
    temp: 0.0,
    seed: 0,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/gguf_llm/generate.rb [options]"
    opts.on("--gguf PATH", String, "GGUF file to load (or download)") { |v| options[:gguf] = v }
    opts.on("--repo NAME", String, "Hugging Face repo if downloading from the Hub") { |v| options[:repo] = v }
    opts.on("--prompt TEXT", String, "Prompt text") { |v| options[:prompt] = v }
    opts.on("--max-tokens N", Integer, "Maximum number of tokens to generate") { |v| options[:max_tokens] = v }
    opts.on("--temp N", Float, "Sampling temperature") { |v| options[:temp] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--python-bin BIN", String, "Python binary for download/tokenizer bridges") { |v| options[:python_bin] = v }
  end
  parser.parse!

  if options[:gguf].nil? || options[:gguf].empty?
    raise ArgumentError, "--gguf is required"
  end

  MLX::Core.random_seed(options[:seed])
  model, tokenizer = GGUFLLM.load(options[:gguf], options[:repo], python_bin: options[:python_bin])
  GGUFLLM.run_generate(
    model: model,
    tokenizer: tokenizer,
    prompt: options[:prompt],
    max_tokens: options[:max_tokens],
    temp: options[:temp]
  )
end
