# frozen_string_literal: true

require "optparse"

require_relative "main"
require_relative "../../benchmark/parity"

if $PROGRAM_NAME == __FILE__
  options = { seed: 11 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby mnist/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])
  rng = Random.new(options[:seed])

  train_x, train_y, test_x, test_y = MnistExample::Dataset.synthetic_mnist(
    train_size: 128,
    test_size: 32,
    seed: options[:seed]
  )
  raise "Train image shape mismatch" unless train_x.shape == [128, 784]
  raise "Train label shape mismatch" unless train_y.shape == [128]
  raise "Test image shape mismatch" unless test_x.shape == [32, 784]
  raise "Test label shape mismatch" unless test_y.shape == [32]

  model = MnistExample::MLP.new(num_layers: 2, input_dim: 784, hidden_dim: 32, output_dim: 10)
  ids = MLX::Core.array((0...16).to_a, MLX::Core.int32)
  sample_x = MLX::Core.take(train_x, ids, 0)
  sample_y = MLX::Core.take(train_y, ids, 0)

  logits = model.call(sample_x)
  MLX::Core.eval(logits)
  raise "Forward shape mismatch: #{logits.shape.inspect}" unless logits.shape == [16, 10]

  optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
  loss_and_grad_fn = MLX::NN.value_and_grad(
    model,
    ->(x, y) { MnistExample::Train.loss_fn(model, x, y) }
  )

  before = MLX::Core.array(model.layers.first.weight.to_a, model.layers.first.weight.dtype)
  loss, grads = loss_and_grad_fn.call(sample_x, sample_y)
  optimizer.update(model, grads)
  MLX::Core.eval(loss, model.parameters, optimizer.state)
  raise "Training loss is not finite" unless loss.item.finite?

  after = model.layers.first.weight
  delta = MLX::Core.sum(MLX::Core.abs(MLX::Core.subtract(after, before)))
  MLX::Core.eval(delta)
  raise "Optimizer step did not update first layer weights" if delta.item <= 0.0

  epoch_loss = MnistExample::Train.train_epoch(
    model,
    train_x,
    train_y,
    batch_size: 32,
    loss_and_grad_fn: loss_and_grad_fn,
    optimizer: optimizer,
    rng: rng
  )
  raise "Epoch loss is not finite" unless epoch_loss.finite?

  acc = MnistExample::Train.accuracy(model, test_x, test_y)
  raise "Accuracy out of range: #{acc}" unless acc >= 0.0 && acc <= 1.0

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "mnist",
      inputs: { sample_x: sample_x },
      outputs: { logits: logits },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
  end

  puts "Tests pass :)"
end
