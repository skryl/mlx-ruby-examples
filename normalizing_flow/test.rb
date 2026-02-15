# frozen_string_literal: true

require "optparse"
require "tmpdir"

require_relative "main"

if $PROGRAM_NAME == __FILE__
  options = { seed: 31 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby normalizing_flow/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  normal = NormalizingFlowExample::Normal.new(
    MLX::Core.zeros([2]),
    MLX::Core.ones([2])
  )
  sample = normal.sample([8, 2])
  log_prob = normal.log_prob(sample)
  MLX::Core.eval(sample, log_prob)
  raise "Normal sample shape mismatch" unless sample.shape == [8, 2]
  raise "Normal log_prob shape mismatch" unless log_prob.shape == [8, 2]

  params = MLX::Core.random_uniform([8, 4], -0.1, 0.1, MLX::Core.float32)
  x = MLX::Core.random_uniform([8, 2], -1.0, 1.0, MLX::Core.float32)
  affine = NormalizingFlowExample::AffineBijector.new(params)
  y, ldj_f = affine.forward_and_log_det(x)
  x2, ldj_i = affine.inverse_and_log_det(y)
  MLX::Core.eval(y, ldj_f, x2, ldj_i)
  max_err = MLX::Core.max(MLX::Core.abs(MLX::Core.subtract(x2, x)))
  MLX::Core.eval(max_err)
  raise "Affine inverse mismatch: #{max_err.item}" if max_err.item > 1e-4

  model = NormalizingFlowExample::RealNVP.new(4, 2, 32, 2)
  raise "RealNVP missing DSL trainer helper" unless model.respond_to?(:trainer)
  batch = MLX::Core.random_uniform([16, 2], -2.0, 2.0, MLX::Core.float32)
  log_density = model.log_prob(batch)
  MLX::Core.eval(log_density)
  raise "RealNVP log_prob shape mismatch" unless log_density.shape == [16]

  optimizer = MLX::Optimizers::Adam.new(learning_rate: 1e-3)
  loss_and_grad_fn = MLX::NN.value_and_grad(
    model,
    ->(input) { NormalizingFlowExample::Train.loss_fn(model, input) }
  )

  before = MLX::Core.array(model.conditioner_list.first.layers.first.weight.to_a, model.conditioner_list.first.layers.first.weight.dtype)
  loss, grads = loss_and_grad_fn.call(batch)
  optimizer.update(model, grads)
  MLX::Core.eval(loss, model.parameters, optimizer.state)
  raise "Training loss is not finite" unless loss.item.finite?

  after = model.conditioner_list.first.layers.first.weight
  delta = MLX::Core.sum(MLX::Core.abs(MLX::Core.subtract(after, before)))
  MLX::Core.eval(delta)
  raise "Optimizer step did not update conditioner weights" if delta.item <= 0.0

  sample2 = model.sample([32, 2])
  MLX::Core.eval(sample2)
  raise "Sample shape mismatch" unless sample2.shape == [32, 2]

  Dir.mktmpdir("nf-test-") do |dir|
    out_file = File.join(dir, "samples.npz")
    NormalizingFlowExample::Train.run(
      n_steps: 2,
      n_batch: 8,
      n_transforms: 2,
      d_params: 2,
      d_hidden: 16,
      n_layers: 2,
      learning_rate: 1e-3,
      noise: 0.06,
      cpu: true,
      seed: options[:seed],
      n_samples: 128,
      n_plot_samples: 64,
      output: out_file,
      report_every: 1
    )
    raise "Missing samples output" unless File.exist?(out_file)
    keys = MLX::Core.load(out_file).to_a.map { |k, _v| k.to_s }
    raise "Missing transform_0 in output" unless keys.include?("transform_0")
    raise "Missing original in output" unless keys.include?("original")
  end

  puts "Tests pass :)"
end
