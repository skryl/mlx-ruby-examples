# frozen_string_literal: true

require "optparse"

require_relative "dataset"
require_relative "resnet"

if $PROGRAM_NAME == __FILE__
  options = { seed: 17 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby cifar/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  model = CifarExample.resnet20
  x = MLX::Core.random_uniform([4, 32, 32, 3], -1.0, 1.0, MLX::Core.float32)
  logits = model.call(x)
  MLX::Core.eval(logits)
  unless logits.shape == [4, 10]
    raise "Forward shape mismatch: expected [4, 10], got #{logits.shape.inspect}"
  end

  block = CifarExample::Block.new(16, 32, stride: 2)
  block_input = MLX::Core.random_uniform([2, 32, 32, 16], -1.0, 1.0, MLX::Core.float32)
  block_out = block.call(block_input)
  MLX::Core.eval(block_out)
  unless block_out.shape == [2, 16, 16, 32]
    raise "Block stride-2 shape mismatch: expected [2, 16, 16, 32], got #{block_out.shape.inspect}"
  end

  train_data, test_data = CifarExample::Dataset.get_cifar10(
    8,
    synthetic: true,
    train_samples: 32,
    test_samples: 16,
    seed: options[:seed]
  )
  first_train = train_data.first
  first_test = test_data.first
  unless first_train.fetch("image").shape == [8, 32, 32, 3]
    raise "Train batch image shape mismatch"
  end
  unless first_train.fetch("label").shape == [8]
    raise "Train batch label shape mismatch"
  end
  unless first_test.fetch("image").shape == [8, 32, 32, 3]
    raise "Test batch image shape mismatch"
  end

  optimizer = MLX::Optimizers::Adam.new(learning_rate: 1e-3)
  train_step = MLX::NN.value_and_grad(
    model,
    lambda do |inp, tgt|
      out = model.call(inp)
      loss = MLX::Core.mean(MLX::NN::Losses.cross_entropy(out, tgt))
      acc = MLX::Core.mean(MLX::Core.equal(MLX::Core.argmax(out, -1), tgt))
      [loss, acc]
    end
  )

  before = MLX::Core.array(model.conv1.weight.to_a, model.conv1.weight.dtype)
  (loss, acc), grads = train_step.call(first_train.fetch("image"), first_train.fetch("label"))
  optimizer.update(model, grads)
  MLX::Core.eval(loss, acc, model.parameters, optimizer.state)

  unless loss.item.finite?
    raise "Training loss is not finite"
  end
  unless acc.item.finite?
    raise "Training accuracy is not finite"
  end

  after = model.conv1.weight
  delta = MLX::Core.sum(MLX::Core.abs(MLX::Core.subtract(after, before)))
  MLX::Core.eval(delta)
  if delta.item <= 0.0
    raise "Optimizer step did not update conv1 weights"
  end

  train_data.reset
  test_data.reset
  unless train_data.first.fetch("image").shape == [8, 32, 32, 3]
    raise "Iterator reset failed for train data"
  end
  unless test_data.first.fetch("image").shape == [8, 32, 32, 3]
    raise "Iterator reset failed for test data"
  end

  puts "Tests pass :)"
end
