# frozen_string_literal: true

require "optparse"

require_relative "llama"
require_relative "../../../benchmark/parity"

module LlamaExample
  module TestHelpers
    module_function

    def last_logits(logits)
      index = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
      step = MLX::Core.take(logits, index, 1)
      MLX::Core.squeeze(step, 1)
    end

    def token_id(token)
      value = token.to_a
      value = value[0] if value.is_a?(Array)
      value.to_i
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { seed: 7 }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/llama/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  args = LlamaExample::ModelArgs.new(
    dim: 16,
    n_layers: 2,
    head_dim: 8,
    hidden_dim: 32,
    n_heads: 2,
    n_kv_heads: 2,
    norm_eps: 1e-5,
    vocab_size: 64,
    rope_theta: 10_000,
    rope_traditional: true
  )

  model = LlamaExample::Llama.new(args)
  prompt = MLX::Core.array([[1, 2, 3, 4]], MLX::Core.int32)

  logits = model.call(prompt)
  MLX::Core.eval(logits)
  unless logits.shape == [1, 4, args.vocab_size]
    raise "Forward shape mismatch: expected [1, 4, #{args.vocab_size}], got #{logits.shape.inspect}"
  end

  generator = model.generate(prompt, temp: 0.0)
  token1 = generator.next
  MLX::Core.eval(token1)

  expected1 = MLX::Core.argmax(LlamaExample::TestHelpers.last_logits(logits), -1)
  MLX::Core.eval(expected1)
  unless LlamaExample::TestHelpers.token_id(token1) == LlamaExample::TestHelpers.token_id(expected1)
    raise "First generated token mismatch against greedy forward argmax"
  end

  prompt2 = MLX::Core.array([[1, 2, 3, 4, LlamaExample::TestHelpers.token_id(token1)]], MLX::Core.int32)
  logits2 = model.call(prompt2)
  MLX::Core.eval(logits2)
  expected2 = MLX::Core.argmax(LlamaExample::TestHelpers.last_logits(logits2), -1)
  MLX::Core.eval(expected2)

  token2 = generator.next
  MLX::Core.eval(token2)
  unless LlamaExample::TestHelpers.token_id(token2) == LlamaExample::TestHelpers.token_id(expected2)
    raise "Second generated token mismatch: cache path diverged from full forward"
  end

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "llms/llama",
      inputs: { prompt: prompt },
      outputs: { logits: logits },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
  end

  puts "Tests pass :)"
end
