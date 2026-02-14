# frozen_string_literal: true

require "optparse"
require "tmpdir"

require_relative "main"

if $PROGRAM_NAME == __FILE__
  options = { seed: 13 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby cvae/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  train_iter, test_iter = CvaeExample::Dataset.mnist(
    batch_size: 8,
    img_size: [64, 64],
    synthetic: true,
    train_size: 64,
    test_size: 32,
    seed: options[:seed]
  )

  train_batch = train_iter.first
  test_batch = test_iter.first
  raise "Train batch image shape mismatch" unless train_batch.fetch("image").shape == [8, 64, 64, 1]
  raise "Train batch label shape mismatch" unless train_batch.fetch("label").shape == [8]
  raise "Test batch image shape mismatch" unless test_batch.fetch("image").shape == [8, 64, 64, 1]
  raise "Test batch label shape mismatch" unless test_batch.fetch("label").shape == [8]

  model = CvaeExample::CVAE.new(4, [64, 64, 1], 32)
  x_recon, mu, logvar = model.call(train_batch.fetch("image"))
  MLX::Core.eval(x_recon, mu, logvar)
  raise "Reconstruction shape mismatch" unless x_recon.shape == [8, 64, 64, 1]
  raise "Mu shape mismatch" unless mu.shape == [8, 4]
  raise "Logvar shape mismatch" unless logvar.shape == [8, 4]

  loss = CvaeExample::Train.loss_fn(model, train_batch.fetch("image"))
  MLX::Core.eval(loss)
  raise "Loss is not finite" unless loss.item.finite?

  optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3)
  loss_and_grad_fn = MLX::NN.value_and_grad(model, ->(x) { CvaeExample::Train.loss_fn(model, x) })

  before = MLX::Core.array(model.encoder.conv1.weight.to_a, model.encoder.conv1.weight.dtype)
  step_loss, grads = loss_and_grad_fn.call(train_batch.fetch("image"))
  optimizer.update(model, grads)
  MLX::Core.eval(step_loss, model.parameters, optimizer.state)
  raise "Step loss is not finite" unless step_loss.item.finite?

  after = model.encoder.conv1.weight
  delta = MLX::Core.sum(MLX::Core.abs(MLX::Core.subtract(after, before)))
  MLX::Core.eval(delta)
  raise "Optimizer step did not update encoder weights" if delta.item <= 0.0

  latent = model.encode(test_batch.fetch("image"))
  decoded = model.decode(latent)
  MLX::Core.eval(decoded)
  raise "Decode shape mismatch" unless decoded.shape == [8, 64, 64, 1]

  Dir.mktmpdir("cvae-test-") do |dir|
    recon_file = File.join(dir, "recon.pgm")
    sample_file = File.join(dir, "sample.pgm")
    CvaeExample::Train.reconstruct(model, train_batch, recon_file)
    CvaeExample::Train.generate(model, sample_file, num_samples: 16)
    raise "Missing reconstruction output" unless File.exist?(recon_file)
    raise "Missing sample output" unless File.exist?(sample_file)
  end

  puts "Tests pass :)"
end
