# frozen_string_literal: true

require "optparse"

require_relative "mistral"
require_relative "../../benchmark/parity"
module MistralExample
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
  options = {
    seed: 7,
    integration_model_path: nil,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/mistral/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--integration-model-path PATH", String, "Run optional checkpoint integration test") do |v|
      options[:integration_model_path] = v
    end
    opts.on("--python-bin BIN", String, "Python binary for sentencepiece bridge") { |v| options[:python_bin] = v }
  end
  parser.parse!
  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  vocab_size = 100
  length = 32
  args = MistralExample::ModelArgs.new(
    dim: 128,
    n_layers: 2,
    head_dim: 32,
    hidden_dim: 256,
    n_heads: 4,
    n_kv_heads: 4,
    norm_eps: 1e-3,
    vocab_size: vocab_size
  )

  model = MistralExample::Mistral.new(args)
  inputs = MLX::Core.array([Array.new(length) { |i| i % vocab_size }], MLX::Core.int32)
  logits, cache = model.call(inputs)
  MLX::Core.eval(logits)

  unless logits.shape == [1, length, vocab_size]
    raise "Model output shape mismatch: expected [1, #{length}, #{vocab_size}], got #{logits.shape.inspect}"
  end
  unless logits.dtype == MLX::Core.float32
    raise "Model output dtype mismatch: expected float32, got #{logits.dtype}"
  end
  unless cache.length == args.n_layers
    raise "Cache length mismatch: expected #{args.n_layers}, got #{cache.length}"
  end

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "llms/mistral",
      inputs: { inputs: inputs },
      outputs: { logits: logits },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
    puts "Tests pass :)"
    exit 0
  end

  params = MLX::Utils.tree_map(
    lambda { |p| p.astype(MLX::Core.float16) },
    model.parameters
  )
  model.update(params)
  logits_fp16, = model.call(inputs)
  MLX::Core.eval(logits_fp16)
  unless logits_fp16.dtype == MLX::Core.float16
    raise "Model float16 conversion mismatch: expected float16, got #{logits_fp16.dtype}"
  end

  prompt = MLX::Core.array([1, 2, 3, 4], MLX::Core.int32)
  generator = MistralExample.generate(prompt, model, temp: 0.0)
  token1 = generator.next
  MLX::Core.eval(token1)

  logits_prompt, = model.call(MLX::Core.expand_dims(prompt, 0))
  expected1 = MLX::Core.argmax(MistralExample.last_logits(logits_prompt), -1)
  MLX::Core.eval(expected1)
  unless MistralExample::TestHelpers.token_id(token1) == MistralExample::TestHelpers.token_id(expected1)
    raise "First generated token mismatch against greedy forward argmax"
  end

  prompt_plus_1 = MLX::Core.array([1, 2, 3, 4, MistralExample::TestHelpers.token_id(token1)], MLX::Core.int32)
  logits_prompt2, = model.call(MLX::Core.expand_dims(prompt_plus_1, 0))
  expected2 = MLX::Core.argmax(MistralExample.last_logits(logits_prompt2), -1)
  MLX::Core.eval(expected2)

  token2 = generator.next
  MLX::Core.eval(token2)
  unless MistralExample::TestHelpers.token_id(token2) == MistralExample::TestHelpers.token_id(expected2)
    raise "Second generated token mismatch: cache path diverged from full forward"
  end

  if !options[:integration_model_path].nil? && Dir.exist?(options[:integration_model_path])
    model, tokenizer = MistralExample.load_model(
      options[:integration_model_path],
      python_bin: options[:python_bin]
    )
    prompt_tokens = MLX::Core.array(tokenizer.encode("This is a test"), MLX::Core.int32)
    generated = []
    MistralExample.generate(prompt_tokens, model).each do |token|
      generated << token
      break if generated.length >= 30
    end
    MLX::Core.eval(*generated)
    expected = [
      302, 272, 11_843, 11_837, 1587, 28_723, 851, 349, 865, 264,
      1369, 28_723, 13, 13, 3381, 456, 654, 264, 1353, 11_843,
      28_725, 368, 682, 347, 2240, 767, 298, 511, 28_723, 13
    ]
    actual = generated.map { |t| t.item.to_i }
    unless actual == expected
      raise "Integration token mismatch for mistral-7B-v0.1 checkpoint"
    end
  end

  puts "Tests pass :)"
end
