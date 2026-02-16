# frozen_string_literal: true

require "optparse"
require "tmpdir"

require_relative "main"
require_relative "../benchmark/parity"
if $PROGRAM_NAME == __FILE__
  options = { seed: 23 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby transformer_lm/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!
  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  Dir.mktmpdir("transformer-lm-ds-") do |dir|
    File.write(File.join(dir, "train.txt"), "hello world\nmlx ruby\n")
    File.write(File.join(dir, "valid.txt"), "hello mlx\n")
    File.write(File.join(dir, "test.txt"), "ruby world\n")

    vocab, train_data, valid_data, test_data = TransformerLmExample::Datasets._load(
      dir,
      ["train.txt", "valid.txt", "test.txt"]
    )
    raise "Expected <eos> in vocab" unless vocab.key?("<eos>")
    raise "Expected non-empty train ids" unless train_data.shape[0] > 0
    raise "Expected non-empty valid ids" unless valid_data.shape[0] > 0
    raise "Expected non-empty test ids" unless test_data.shape[0] > 0
  end

  vocab, train, valid, = TransformerLmExample::Train.synthetic_dataset(
    total_tokens: 4096,
    vocab_size: 128,
    seed: options[:seed]
  )

  model = TransformerLmExample::TransformerLM.new(
    vocab_size: vocab.length,
    num_layers: 2,
    dims: 32,
    num_heads: 4,
    checkpoint: false
  )

  samples = TransformerLmExample::Train.to_samples(15, train)
  raise "to_samples shape mismatch" unless samples.shape[1] == 16

  iterator = TransformerLmExample::Train.iterate_batches(4, 15, train, seed: options[:seed])
  batch = iterator.next
  raise "Batch shape mismatch: #{batch.shape.inspect}" unless batch.shape == [4, 16]

  x = MLX::Core.slice(batch, [0, 0], [batch.shape[0], batch.shape[1] - 1])
  logits = model.call(x)
  MLX::Core.eval(logits)
  raise "Forward logits shape mismatch: #{logits.shape.inspect}" unless logits.shape == [4, 15, vocab.length]

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "transformer_lm",
      inputs: { x: x },
      outputs: { logits: logits },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
    puts "Tests pass :)"
    exit 0
  end

  loss = TransformerLmExample::Train.loss_fn(model, batch)
  MLX::Core.eval(loss)
  raise "Loss is not finite" unless loss.item.finite?

  optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3, weight_decay: 0.0)
  loss_and_grad_fn = MLX::NN.value_and_grad(model, ->(inputs) { TransformerLmExample::Train.loss_fn(model, inputs) })

  before = MLX::Core.array(model.out_proj.weight.to_a, model.out_proj.weight.dtype)
  step_loss, grads = loss_and_grad_fn.call(batch)
  optimizer.update(model, grads)
  MLX::Core.eval(step_loss, model.parameters, optimizer.state)
  raise "Step loss is not finite" unless step_loss.item.finite?

  delta = MLX::Core.sum(
    MLX::Core.abs(
      MLX::Core.subtract(model.out_proj.weight, before)
    )
  )
  MLX::Core.eval(delta)
  raise "Optimizer step did not update output projection weights" if delta.item <= 0.0

  val_loss = TransformerLmExample::Train.eval_fn(model, valid, context_size: 15, batch_size: 4)
  raise "Validation loss is not finite" unless val_loss.finite?

  puts "Tests pass :)"
end
