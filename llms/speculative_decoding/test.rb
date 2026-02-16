# frozen_string_literal: true

require "optparse"

require_relative "model"
require_relative "../../benchmark/parity"
module SpeculativeDecodingExample
  module Parity
    module_function

    def allclose?(left, right, rtol:, atol:)
      lhs = flatten_numeric(left)
      rhs = flatten_numeric(right)
      return false unless lhs.length == rhs.length

      lhs.each_with_index do |a, i|
        b = rhs[i]
        tolerance = atol + (rtol * b.abs)
        return false if (a - b).abs > tolerance
      end
      true
    end

    def flatten_numeric(value, out = [])
      if value.is_a?(Array)
        value.each { |item| flatten_numeric(item, out) }
      else
        out << value.to_f
      end
      out
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    seed: 5,
    rtol: 1e-4,
    atol: 1e-5
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/speculative_decoding/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--rtol N", Float, "Relative tolerance") { |v| options[:rtol] = v }
    opts.on("--atol N", Float, "Absolute tolerance") { |v| options[:atol] = v }
  end
  parser.parse!
  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  config = SpeculativeDecodingExample::T5Config.new(
    "d_model" => 32,
    "d_kv" => 8,
    "d_ff" => 64,
    "num_heads" => 4,
    "num_layers" => 2,
    "num_decoder_layers" => 2,
    "layer_norm_epsilon" => 1e-6,
    "relative_attention_num_buckets" => 8,
    "relative_attention_max_distance" => 32,
    "feed_forward_proj" => "relu",
    "tie_word_embeddings" => true,
    "vocab_size" => 128
  )

  model = SpeculativeDecodingExample::Model.new(config)
  inputs = MLX::Core.array([[1, 2, 3, 4, 5, 6]], MLX::Core.int32)
  decoder_inputs = MLX::Core.array([[0, 7, 8]], MLX::Core.int32)
  output = model.call(inputs, decoder_inputs)
  MLX::Core.eval(output)
  unless output.shape == [decoder_inputs.shape[1], config.vocab_size]
    raise "Model call shape mismatch: expected [#{decoder_inputs.shape[1]}, #{config.vocab_size}], got #{output.shape.inspect}"
  end

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "llms/speculative_decoding",
      inputs: { inputs: inputs, decoder_inputs: decoder_inputs },
      outputs: { output: output },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
    puts "Tests pass :)"
    exit 0
  end

  memory = model.encode(inputs)
  model.reset_cache
  full_logits = model.decode(decoder_inputs, memory)
  MLX::Core.eval(full_logits)
  full_last = MLX::Core.take(full_logits, MLX::Core.array([decoder_inputs.shape[1] - 1], MLX::Core.int32), 1)
  full_last = MLX::Core.squeeze(full_last, 1)

  model.reset_cache
  model.decode(MLX::Core.array([[0, 7]], MLX::Core.int32), memory)
  step_logits = model.decode(MLX::Core.array([[8]], MLX::Core.int32), memory)
  MLX::Core.eval(step_logits)
  step_last = MLX::Core.squeeze(step_logits, 1)

  unless SpeculativeDecodingExample::Parity.allclose?(
    full_last.to_a,
    step_last.to_a,
    rtol: options[:rtol],
    atol: options[:atol]
  )
    raise "Decoder cache mismatch: incremental decode diverged from full decode"
  end

  cache_len = model.cache[0][0].shape[2]
  model.truncate_cache(1)
  unless model.cache[0][0].shape[2] == (cache_len - 1)
    raise "Cache truncate mismatch: expected #{cache_len - 1}, got #{model.cache[0][0].shape[2]}"
  end
  model.truncate_cache(cache_len)
  unless model.cache[0].nil?
    raise "Cache truncate reset mismatch: cache should be reset to nil entries"
  end

  puts "Tests pass :)"
end
