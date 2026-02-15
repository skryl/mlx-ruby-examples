# frozen_string_literal: true

require "optparse"
require "time"

require_relative "mnist"

module MnistExample
  class MLP < MLX::DSL::Model
    option :num_layers
    option :input_dim
    option :hidden_dim
    option :output_dim

    layer :network do
      layer_sizes = [input_dim] + Array.new(num_layers, hidden_dim) + [output_dim]
      MLX::NN::Sequential.new(
        *layer_sizes.each_cons(2).map { |in_dim, out_dim| MLX::NN::Linear.new(in_dim, out_dim) }
      )
    end

    def layers
      network.layers
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

    def train_data_pipeline(x, y, batch_size:, seed:)
      MLX::DSL::Data
        .from(0...y.shape[0])
        .shuffle(seed: seed)
        .batch(batch_size)
        .map do |batch_ids|
          ids = MLX::Core.array(batch_ids, MLX::Core.int32)
          [
            MLX::Core.take(x, ids, 0),
            MLX::Core.take(y, ids, 0)
          ]
        end
    end

    def run(options)
      unless DATASETS.include?(options[:dataset])
        raise ArgumentError, "--dataset must be one of #{DATASETS.join(', ')}"
      end

      MLX::Core.set_default_device(MLX::Core.cpu) if options[:cpu]
      MLX::Core.random_seed(options[:seed])

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
      trainer = model.trainer(optimizer: optimizer) do |x:, y:|
        loss_fn(model, x, y)
      end

      epoch_started_at = {}
      trainer.before_epoch do |ctx|
        epoch_started_at[ctx.fetch(:epoch)] = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      end
      trainer.after_epoch do |ctx|
        epoch = ctx.fetch(:epoch)
        started_at = epoch_started_at.fetch(epoch, Process.clock_gettime(Process::CLOCK_MONOTONIC))
        test_acc = accuracy(model, test_images, test_labels)
        puts format(
          "Epoch %d: Train loss %.4f | Test accuracy %.3f | Time %.3f (s)",
          epoch,
          ctx.fetch(:epoch_loss).to_f,
          test_acc,
          Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at
        )
      end

      train_data = lambda do |epoch:, **_kwargs|
        train_data_pipeline(
          train_images,
          train_labels,
          batch_size: options[:batch_size],
          seed: options[:seed] + epoch.to_i
        )
      end
      trainer.register_dataflow(
        :mnist_train,
        train: {
          collate: :xy,
          reduce: :mean
        }
      )
      split_plan = MLX::DSL.splits do
        train(train_data)
      end

      run_bundle_path = options[:run_bundle_path]
      resume_source = options[:resume_from]
      artifact_kwargs = {}
      artifact_kwargs[:resume] = resume_source unless resume_source.nil? || resume_source.empty?
      unless run_bundle_path.nil? || run_bundle_path.empty?
        artifact_kwargs[:run_bundle] = {
          enabled: true,
          path: run_bundle_path,
          config: {
            "example" => "mnist",
            "dataset" => options[:dataset]
          }
        }
      end
      trainer.artifact_policy(**artifact_kwargs) unless artifact_kwargs.empty?

      exp = MLX::DSL.experiment("mnist") do
        trainer(trainer)
        data(
          train: split_plan,
          **trainer.use_dataflow(:mnist_train),
          epochs: options[:epochs],
          keep_losses: false,
          strict_data_reuse: true
        )
      end
      exp.report

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
    run_bundle_path: nil,
    resume_from: nil,
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
    opts.on("--run-bundle PATH", String, "Auto-save DSL run bundle path") { |v| options[:run_bundle_path] = v }
    opts.on("--resume-from SOURCE", String, "Resume source (checkpoint or run bundle path)") { |v| options[:resume_from] = v }
    opts.on("--synthetic", "Use synthetic data instead of downloading MNIST") { options[:synthetic] = true }
    opts.on("--train-size N", Integer, "Synthetic train size") { |v| options[:train_size] = v }
    opts.on("--test-size N", Integer, "Synthetic test size") { |v| options[:test_size] = v }
  end
  parser.parse!

  MnistExample::Train.run(options)
end
