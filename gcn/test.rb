# frozen_string_literal: true

require "optparse"

require_relative "main"
require_relative "../benchmark/parity"
if $PROGRAM_NAME == __FILE__
  options = { seed: 19 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby gcn/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!
  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  x, y, adj = GcnExample::Datasets.load_data(
    synthetic: true,
    synthetic_nodes: 128,
    feature_dim: 32,
    classes: 7,
    edge_prob: 0.05,
    seed: options[:seed]
  )
  raise "Feature shape mismatch" unless x.shape == [128, 32]
  raise "Label shape mismatch" unless y.shape == [128]
  raise "Adjacency shape mismatch" unless adj.shape == [128, 128]

  train_mask, val_mask, test_mask = GcnExample::Datasets.train_val_test_mask(num_nodes: x.shape[0])
  raise "Train mask empty" if train_mask.shape[0].zero?
  raise "Val mask empty" if val_mask.shape[0].zero?
  raise "Test mask empty" if test_mask.shape[0].zero?

  model = GcnExample::GCN.new(
    x_dim: x.shape[-1],
    h_dim: 16,
    out_dim: 7,
    nb_layers: 2,
    dropout: benchmark_enabled ? 0.0 : 0.2,
    bias: true
  )
  raise "GCN missing DSL trainer helper" unless model.respond_to?(:trainer)
  BenchmarkDeterministic.reinitialize_module!(model) if benchmark_enabled
  logits = model.call(x, adj)
  MLX::Core.eval(logits)
  raise "Forward shape mismatch: #{logits.shape.inspect}" unless logits.shape == [128, 7]

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "gcn",
      inputs: { x: x, adj: adj },
      outputs: { logits: logits },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
    puts "Tests pass :)"
    exit 0
  end

  train_logits = GcnExample::Train.select_rows(logits, train_mask)
  train_labels = GcnExample::Train.select_rows(y, train_mask)
  loss = GcnExample::Train.loss_fn(train_logits, train_labels, weight_decay: 1e-4, parameters: model.parameters)
  MLX::Core.eval(loss)
  raise "Loss is not finite" unless loss.item.finite?

  optimizer = MLX::Optimizers::Adam.new(learning_rate: 1e-3)
  loss_and_grad_fn = MLX::NN.value_and_grad(
    model,
    ->(features, graph_adj, labels, mask) { GcnExample::Train.forward_loss(model, features, graph_adj, labels, mask, 0.0) }
  )

  before = MLX::Core.array(model.gcn_layers.first.linear.weight.to_a, model.gcn_layers.first.linear.weight.dtype)
  step_loss, grads = loss_and_grad_fn.call(x, adj, y, train_mask)
  optimizer.update(model, grads)
  MLX::Core.eval(step_loss, model.parameters, optimizer.state)
  raise "Step loss is not finite" unless step_loss.item.finite?

  delta = MLX::Core.sum(
    MLX::Core.abs(
      MLX::Core.subtract(model.gcn_layers.first.linear.weight, before)
    )
  )
  MLX::Core.eval(delta)
  raise "Optimizer step did not update GCN weights" if delta.item <= 0.0

  GcnExample::Train.run(
    hidden_dim: 16,
    dropout: 0.2,
    nb_layers: 1,
    nb_classes: 7,
    bias: true,
    lr: 1e-3,
    weight_decay: 0.0,
    patience: 3,
    epochs: 2,
    cpu: true,
    seed: options[:seed],
    synthetic: true,
    synthetic_nodes: 128,
    synthetic_feature_dim: 32,
    synthetic_edge_prob: 0.05,
    data_root: "gcn/data",
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  )

  puts "Tests pass :)"
end
