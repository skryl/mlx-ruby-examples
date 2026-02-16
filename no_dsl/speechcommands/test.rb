# frozen_string_literal: true

require "optparse"

require_relative "main"
require_relative "../../benchmark/parity"

if $PROGRAM_NAME == __FILE__
  options = { seed: 29 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby speechcommands/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  # Synthetic iterator contract
  train_iter = SpeechcommandsExample::Dataset.prepare_dataset(
    batch_size: 8,
    split: "train",
    synthetic: true,
    input_res: [98, 40],
    num_classes: 12,
    train_samples: 32,
    val_samples: 16,
    test_samples: 16,
    seed: options[:seed]
  )
  batch = train_iter.first
  raise "Audio batch shape mismatch" unless batch.fetch("audio").shape == [8, 98, 40, 1]
  raise "Label batch shape mismatch" unless batch.fetch("label").shape == [8]

  # Small custom model forward checks
  model = SpeechcommandsExample::KWT.new(
    [98, 40],
    [1, 40],
    12,
    dim: 32,
    depth: 2,
    heads: 2,
    mlp_dim: 64,
    emb_dropout: 0.0
  )
  BenchmarkDeterministic.reinitialize_module!(model) if benchmark_enabled
  x = MLX::Core.normal([4, 98, 40, 1])
  y = model.call(x)
  MLX::Core.eval(y)
  raise "Forward shape mismatch (4D input): #{y.shape.inspect}" unless y.shape == [4, 12]

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "speechcommands",
      inputs: { x: x },
      outputs: { y: y },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
    puts "Tests pass :)"
    exit 0
  end

  x3 = MLX::Core.normal([4, 98, 40])
  y3 = model.call(x3)
  MLX::Core.eval(y3)
  raise "Forward shape mismatch (3D input): #{y3.shape.inspect}" unless y3.shape == [4, 12]

  # One training step should update weights
  optimizer = MLX::Optimizers::SGD.new(learning_rate: 1e-3, momentum: 0.9, weight_decay: 1e-4)
  step = MLX::NN.value_and_grad(
    model,
    lambda do |audio, labels|
      logits = model.call(audio)
      loss = MLX::Core.mean(MLX::NN::Losses.cross_entropy(logits, labels))
      acc = MLX::Core.mean(MLX::Core.equal(MLX::Core.argmax(logits, 1), labels))
      [loss, acc]
    end
  )

  before = MLX::Core.array(model.patch_embedding.weight.to_a, model.patch_embedding.weight.dtype)
  (loss, acc), grads = step.call(batch.fetch("audio"), batch.fetch("label"))
  optimizer.update(model, grads)
  MLX::Core.eval(loss, acc, model.parameters, optimizer.state)
  raise "Loss is not finite" unless loss.item.finite?
  raise "Accuracy is not finite" unless acc.item.finite?

  delta = MLX::Core.sum(MLX::Core.abs(MLX::Core.subtract(model.patch_embedding.weight, before)))
  MLX::Core.eval(delta)
  raise "Optimizer step did not update patch embedding weights" if delta.item <= 0.0

  # End-to-end smoke run on tiny synthetic splits
  SpeechcommandsExample::Train.run(
    arch: "kwt1",
    batch_size: 8,
    epochs: 1,
    lr: 1e-3,
    seed: options[:seed],
    cpu: true,
    synthetic: true,
    data_file: nil,
    num_classes: 12,
    input_res: [98, 40],
    patch_res: [1, 40],
    dropout: 0.0,
    emb_dropout: 0.1,
    train_samples: 32,
    val_samples: 16,
    test_samples: 16,
    best_ckpt: nil
  )

  puts "Tests pass :)"
end
