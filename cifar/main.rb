# frozen_string_literal: true

require "optparse"
require "time"

require_relative "dataset"
require_relative "resnet"

module CifarExample
  module Train
    module_function

    ARCHES = %w[resnet20 resnet32 resnet44 resnet56 resnet110 resnet1202].freeze

    def print_zero(group, *args)
      return if group.rank != 0

      puts(*args)
    end

    def eval_fn(model, inp, tgt)
      logits = model.call(inp)
      preds = MLX::Core.argmax(logits, -1)
      MLX::Core.mean(MLX::Core.equal(preds, tgt))
    end

    def average_stats(world, stats, count)
      return stats.map { |s| s / count.to_f } if world.size == 1

      with_cpu_stream do
        stats_arr = MLX::Core.array(stats, MLX::Core.float32)
        count_arr = MLX::Core.array([count], MLX::Core.float32)
        summed_stats = MLX::Core.all_sum(stats_arr)
        summed_count = MLX::Core.all_sum(count_arr)
        denom = summed_count.item.to_f
        summed_stats.to_a.map { |v| v / denom }
      end
    end

    def with_cpu_stream
      MLX::Core.stream(MLX::Core.cpu) do
        yield
      end
    end

    def train_epoch(model, train_iter, optimizer, epoch, world)
      model.train(true)
      losses = 0.0
      accuracies = 0.0
      samples_per_sec = 0.0
      count = 0

      train_step = MLX::NN.value_and_grad(
        model,
        lambda do |inp, tgt|
          output = model.call(inp)
          loss = MLX::Core.mean(MLX::NN::Losses.cross_entropy(output, tgt))
          acc = MLX::Core.mean(MLX::Core.equal(MLX::Core.argmax(output, -1), tgt))
          [loss, acc]
        end
      )

      train_iter.each_with_index do |batch, batch_counter|
        x = batch.fetch("image")
        y = batch.fetch("label")

        tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        (loss, acc), grads = train_step.call(x, y)
        grads = MLX::NN.average_gradients(grads, world)
        optimizer.update(model, grads)
        MLX::Core.eval(loss, acc, model.parameters, optimizer.state)
        toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)

        losses += loss.item.to_f
        accuracies += acc.item.to_f
        samples_per_sec += x.shape[0] / [toc - tic, 1e-9].max
        count += 1

        next unless (batch_counter % 10).zero?

        l, a, s = average_stats(
          world,
          [losses, accuracies, world.size * samples_per_sec],
          count
        )
        print_zero(
          world,
          format(
            "Epoch %02d [%03d] | Train loss %.3f | Train acc %.3f | Throughput: %.2f images/sec",
            epoch,
            batch_counter,
            l,
            a,
            s
          )
        )
      end

      average_stats(world, [losses, accuracies, world.size * samples_per_sec], count)
    end

    def test_epoch(model, test_iter, world)
      model.eval
      accuracies = 0.0
      count = 0
      test_iter.each do |batch|
        x = batch.fetch("image")
        y = batch.fetch("label")
        acc = eval_fn(model, x, y)
        MLX::Core.eval(acc)
        accuracies += acc.item.to_f
        count += 1
      end

      return accuracies / count.to_f if world.size == 1

      with_cpu_stream do
        acc_arr = MLX::Core.array([accuracies], MLX::Core.float32)
        cnt_arr = MLX::Core.array([count], MLX::Core.float32)
        total_acc = MLX::Core.all_sum(acc_arr).item.to_f
        total_cnt = MLX::Core.all_sum(cnt_arr).item.to_f
        total_acc / total_cnt
      end
    end

    def build_parser(options)
      OptionParser.new do |opts|
        opts.banner = "Usage: ruby cifar/main.rb [options]"
        opts.on("--arch NAME", String, "Model architecture (#{ARCHES.join(', ')})") { |v| options[:arch] = v }
        opts.on("--batch-size N", Integer, "Batch size") { |v| options[:batch_size] = v }
        opts.on("--epochs N", Integer, "Number of epochs") { |v| options[:epochs] = v }
        opts.on("--lr N", Float, "Learning rate") { |v| options[:lr] = v }
        opts.on("--seed N", Integer, "Random seed") { |v| options[:seed] = v }
        opts.on("--cpu", "Use CPU only") { options[:cpu] = true }
        opts.on("--data-root PATH", String, "Directory for CIFAR cache files") { |v| options[:data_root] = v }
        opts.on("--python-bin BIN", String, "Python binary for dataset bridge") { |v| options[:python_bin] = v }
        opts.on("--synthetic", "Use synthetic CIFAR-like data") { options[:synthetic] = true }
        opts.on("--train-samples N", Integer, "Synthetic train samples") { |v| options[:train_samples] = v }
        opts.on("--test-samples N", Integer, "Synthetic test samples") { |v| options[:test_samples] = v }
      end
    end

    def run(options)
      unless ARCHES.include?(options[:arch])
        raise ArgumentError, "--arch must be one of #{ARCHES.join(', ')}"
      end

      MLX::Core.set_default_device(MLX::Core.cpu) if options[:cpu]
      MLX::Core.random_seed(options[:seed])

      world = MLX::Core.init
      if world.size > 1
        puts "Starting rank #{world.rank} of #{world.size}"
      end

      model = CifarExample.public_send(options[:arch])
      print_zero(world, format("Number of params: %.4f M", model.num_params / 1e6))
      optimizer = MLX::Optimizers::Adam.new(learning_rate: options[:lr])

      train_data, test_data = Dataset.get_cifar10(
        options[:batch_size],
        root: options[:data_root],
        synthetic: options[:synthetic],
        train_samples: options[:train_samples],
        test_samples: options[:test_samples],
        seed: options[:seed],
        python_bin: options[:python_bin]
      )

      options[:epochs].times do |epoch|
        tr_loss, tr_acc, throughput = train_epoch(model, train_data, optimizer, epoch, world)
        print_zero(
          world,
          format(
            "Epoch: %d | avg. Train loss %.3f | avg. Train acc %.3f | Throughput: %.2f images/sec",
            epoch,
            tr_loss,
            tr_acc,
            throughput
          )
        )

        test_acc = test_epoch(model, test_data, world)
        print_zero(world, format("Epoch: %d | Test acc %.3f", epoch, test_acc))

        train_data.reset
        test_data.reset
      end
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    arch: "resnet20",
    batch_size: 256,
    epochs: 30,
    lr: 1e-3,
    seed: 0,
    cpu: false,
    data_root: nil,
    python_bin: ENV.fetch("PYTHON_BIN", "python3"),
    synthetic: false,
    train_samples: 4096,
    test_samples: 1024
  }

  parser = CifarExample::Train.build_parser(options)
  parser.parse!
  CifarExample::Train.run(options)
end
