# frozen_string_literal: true

require "optparse"
require "time"

require_relative "mnist"

module MnistExample
  class MLP < MLX::NN::Module
    def initialize(num_layers:, input_dim:, hidden_dim:, output_dim:)
      super()
      layer_sizes = [input_dim] + Array.new(num_layers, hidden_dim) + [output_dim]
      self.layers = layer_sizes.each_cons(2).map { |in_dim, out_dim| MLX::NN::Linear.new(in_dim, out_dim) }
    end

    def call(x)
      hidden = x
      layers[0...-1].each do |layer|
        hidden = MLX::NN.relu(layer.call(hidden))
      end
      layers[-1].call(hidden)
    end
  end

  module Train
    module_function

    DATASETS = %w[mnist fashion_mnist].freeze

    def loss_fn(model, x, y)
      logits = model.call(x)
      MLX::Core.mean(MLX::NN::Losses.cross_entropy(logits, y))
    end

    def accuracy(model, x, y)
      logits = model.call(x)
      preds = MLX::Core.argmax(logits, -1)
      acc = MLX::Core.mean(MLX::Core.equal(preds, y))
      MLX::Core.eval(acc)
      acc.item.to_f
    end

    def batch_iterate(batch_size, x, y, rng:)
      count = y.shape[0]
      order = (0...count).to_a
      order.shuffle!(random: rng)

      Enumerator.new do |enum|
        order.each_slice(batch_size) do |batch_ids|
          ids = MLX::Core.array(batch_ids, MLX::Core.int32)
          batch_x = MLX::Core.take(x, ids, 0)
          batch_y = MLX::Core.take(y, ids, 0)
          enum << [batch_x, batch_y]
        end
      end
    end

    def train_epoch(model, x, y, batch_size:, loss_and_grad_fn:, optimizer:, rng:)
      total_loss = 0.0
      steps = 0

      batch_iterate(batch_size, x, y, rng: rng).each do |batch_x, batch_y|
        loss, grads = loss_and_grad_fn.call(batch_x, batch_y)
        optimizer.update(model, grads)
        MLX::Core.eval(loss, model.parameters, optimizer.state)
        total_loss += loss.item.to_f
        steps += 1
      end

      total_loss / [steps, 1].max.to_f
    end

    def run(options)
      unless DATASETS.include?(options[:dataset])
        raise ArgumentError, "--dataset must be one of #{DATASETS.join(', ')}"
      end

      MLX::Core.set_default_device(MLX::Core.cpu) if options[:cpu]
      MLX::Core.random_seed(options[:seed])
      rng = Random.new(options[:seed])

      train_images, train_labels, test_images, test_labels = Dataset.public_send(
        options[:dataset],
        save_dir: options[:data_root],
        python_bin: options[:python_bin],
        synthetic: options[:synthetic],
        train_size: options[:train_size],
        test_size: options[:test_size],
        seed: options[:seed]
      )

      model = MLP.new(
        num_layers: options[:num_layers],
        input_dim: train_images.shape[-1],
        hidden_dim: options[:hidden_dim],
        output_dim: 10
      )
      optimizer = MLX::Optimizers::SGD.new(learning_rate: options[:learning_rate])
      loss_and_grad_fn = MLX::NN.value_and_grad(model, ->(x, y) { loss_fn(model, x, y) })

      options[:epochs].times do |epoch|
        tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        train_loss = train_epoch(
          model,
          train_images,
          train_labels,
          batch_size: options[:batch_size],
          loss_and_grad_fn: loss_and_grad_fn,
          optimizer: optimizer,
          rng: rng
        )
        test_acc = accuracy(model, test_images, test_labels)
        toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)

        puts format(
          "Epoch %d: Train loss %.4f | Test accuracy %.3f | Time %.3f (s)",
          epoch,
          train_loss,
          test_acc,
          toc - tic
        )
      end

      model
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    cpu: true,
    dataset: "mnist",
    num_layers: 2,
    hidden_dim: 32,
    batch_size: 256,
    epochs: 10,
    learning_rate: 1e-1,
    seed: 0,
    data_root: "/tmp",
    python_bin: ENV.fetch("PYTHON_BIN", "python3"),
    synthetic: false,
    train_size: 60_000,
    test_size: 10_000
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby mnist/main.rb [options]"
    opts.on("--gpu", "Use Metal back-end") { options[:cpu] = false }
    opts.on("--cpu", "Force CPU") { options[:cpu] = true }
    opts.on("--dataset NAME", String, "Dataset: #{MnistExample::Train::DATASETS.join(', ')}") { |v| options[:dataset] = v }
    opts.on("--num-layers N", Integer, "Number of hidden layers") { |v| options[:num_layers] = v }
    opts.on("--hidden-dim N", Integer, "Hidden dimension") { |v| options[:hidden_dim] = v }
    opts.on("--batch-size N", Integer, "Batch size") { |v| options[:batch_size] = v }
    opts.on("--epochs N", Integer, "Epochs") { |v| options[:epochs] = v }
    opts.on("--learning-rate N", Float, "Learning rate") { |v| options[:learning_rate] = v }
    opts.on("--seed N", Integer, "Random seed") { |v| options[:seed] = v }
    opts.on("--data-root PATH", String, "Dataset cache directory") { |v| options[:data_root] = v }
    opts.on("--python-bin BIN", String, "Python binary for bridge script") { |v| options[:python_bin] = v }
    opts.on("--synthetic", "Use synthetic data instead of downloading MNIST") { options[:synthetic] = true }
    opts.on("--train-size N", Integer, "Synthetic train size") { |v| options[:train_size] = v }
    opts.on("--test-size N", Integer, "Synthetic test size") { |v| options[:test_size] = v }
  end
  parser.parse!

  MnistExample::Train.run(options)
end
