# frozen_string_literal: true

require "optparse"

require_relative "mixtral"
require_relative "../../../benchmark/parity"

module MixtralExample
  module TestHelpers
    module_function

    def token_id(token)
      value = token.to_a
      value = value[0] if value.is_a?(Array)
      value.to_i
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { seed: 11 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/mixtral/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  args = MixtralExample::ModelArgs.new(
    dim: 64,
    n_layers: 2,
    head_dim: 16,
    hidden_dim: 128,
    n_heads: 4,
    n_kv_heads: 4,
    norm_eps: 1e-3,
    vocab_size: 97,
    moe: {
      "num_experts_per_tok" => 2,
      "num_experts" => 4
    }
  )

  model = MixtralExample::Mixtral.new(args)
  inputs = MLX::Core.array([Array.new(12) { |i| i % args.vocab_size }], MLX::Core.int32)

  logits, cache = model.call(inputs)
  MLX::Core.eval(logits)
  unless logits.shape == [1, 1, args.vocab_size]
    raise "Mixtral forward shape mismatch: expected [1, 1, #{args.vocab_size}], got #{logits.shape.inspect}"
  end
  unless logits.dtype == MLX::Core.float32
    raise "Mixtral forward dtype mismatch: expected float32, got #{logits.dtype}"
  end
  unless cache.length == args.n_layers
    raise "Mixtral cache length mismatch: expected #{args.n_layers}, got #{cache.length}"
  end

  params = MLX::Utils.tree_map(
    lambda { |p| p.astype(MLX::Core.float16) },
    model.parameters
  )
  model.update(params)
  logits_fp16, = model.call(inputs)
  MLX::Core.eval(logits_fp16)
  unless logits_fp16.dtype == MLX::Core.float16
    raise "Mixtral float16 conversion mismatch: expected float16, got #{logits_fp16.dtype}"
  end

  prompt = MLX::Core.array([1, 2, 3, 4], MLX::Core.int32)
  generator = MixtralExample.generate(prompt, model, temp: 0.0)
  token1 = generator.next
  MLX::Core.eval(token1)

  logits_prompt, = model.call(MLX::Core.expand_dims(prompt, 0))
  expected1 = MLX::Core.argmax(MLX::Core.squeeze(logits_prompt, 1), -1)
  MLX::Core.eval(expected1)
  unless MixtralExample::TestHelpers.token_id(token1) == MixtralExample::TestHelpers.token_id(expected1)
    raise "First generated token mismatch against greedy forward argmax"
  end

  prompt_plus_1 = MLX::Core.array([1, 2, 3, 4, MixtralExample::TestHelpers.token_id(token1)], MLX::Core.int32)
  logits_prompt2, = model.call(MLX::Core.expand_dims(prompt_plus_1, 0))
  expected2 = MLX::Core.argmax(MLX::Core.squeeze(logits_prompt2, 1), -1)
  MLX::Core.eval(expected2)

  token2 = generator.next
  MLX::Core.eval(token2)
  unless MixtralExample::TestHelpers.token_id(token2) == MixtralExample::TestHelpers.token_id(expected2)
    raise "Second generated token mismatch: cache path diverged from full forward"
  end

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "llms/mixtral",
      inputs: { inputs: inputs },
      outputs: { logits: logits },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
  end

  puts "Tests pass :)"
end
