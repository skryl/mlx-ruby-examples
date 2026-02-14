# frozen_string_literal: true

require "optparse"
require "securerandom"
require "tmpdir"
require "time"

require_relative "dataset"
require_relative "kwt"

module SpeechcommandsExample
  module Train
    module_function

    ARCHES = %w[kwt1 kwt2 kwt3].freeze

    def eval_fn(model, x, y)
      logits = model.call(x)
      preds = MLX::Core.argmax(logits, 1)
      MLX::Core.mean(MLX::Core.equal(preds, y))
    end

    def train_epoch(model, train_iter, optimizer, epoch)
      train_step = MLX::NN.value_and_grad(
        model,
        lambda do |x, y|
          output = model.call(x)
          loss = MLX::Core.mean(MLX::NN::Losses.cross_entropy(output, y))
          acc = MLX::Core.mean(MLX::Core.equal(MLX::Core.argmax(output, 1), y))
          [loss, acc]
        end
      )

      losses = []
      accs = []
      samples_per_sec = []

      model.train(true)
      train_iter.reset
      train_iter.each_with_index do |batch, batch_counter|
        x = batch.fetch("audio")
        y = batch.fetch("label")

        tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        (loss, acc), grads = train_step.call(x, y)
        optimizer.update(model, grads)
        MLX::Core.eval(loss, acc, model.parameters, optimizer.state)
        toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)

        loss_val = loss.item.to_f
        acc_val = acc.item.to_f
        throughput = x.shape[0] / [toc - tic, 1e-9].max

        losses << loss_val
        accs << acc_val
        samples_per_sec << throughput

        next unless (batch_counter % 25).zero?

        puts [
          format("Epoch %02d [%03d]", epoch, batch_counter),
          format("Train loss %.3f", loss_val),
          format("Train acc %.3f", acc_val),
          format("Throughput: %.2f samples/second", throughput)
        ].join(" | ")
      end

      [
        MLX::Core.mean(MLX::Core.array(losses, MLX::Core.float32)),
        MLX::Core.mean(MLX::Core.array(accs, MLX::Core.float32)),
        MLX::Core.mean(MLX::Core.array(samples_per_sec, MLX::Core.float32))
      ]
    end

    def test_epoch(model, data_iter)
      model.train(false)
      accuracies = []
      throughput = []

      data_iter.reset
      data_iter.each do |batch|
        x = batch.fetch("audio")
        y = batch.fetch("label")
        tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        acc = eval_fn(model, x, y)
        MLX::Core.eval(acc)
        toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        accuracies << acc.item.to_f
        throughput << x.shape[0] / [toc - tic, 1e-9].max
      end

      [
        MLX::Core.mean(MLX::Core.array(accuracies, MLX::Core.float32)),
        MLX::Core.mean(MLX::Core.array(throughput, MLX::Core.float32))
      ]
    end

    def build_model(options)
      kwargs = {
        input_res: options[:input_res],
        patch_res: options[:patch_res],
        num_classes: options[:num_classes],
        dropout: options[:dropout],
        emb_dropout: options[:emb_dropout]
      }
      SpeechcommandsExample.public_send(options[:arch], **kwargs)
    end

    def run(options)
      unless ARCHES.include?(options[:arch])
        raise ArgumentError, "--arch must be one of #{ARCHES.join(', ')}"
      end

      MLX::Core.set_default_device(MLX::Core.cpu) if options[:cpu]
      MLX::Core.random_seed(options[:seed])

      model = build_model(options)
      puts format("Number of params: %.4f M", model.num_params / 1e6)

      optimizer = MLX::Optimizers::SGD.new(
        learning_rate: options[:lr],
        momentum: 0.9,
        weight_decay: 1e-4
      )

      train_data = Dataset.prepare_dataset(
        batch_size: options[:batch_size],
        split: "train",
        data_file: options[:data_file],
        synthetic: options[:synthetic],
        input_res: options[:input_res],
        num_classes: options[:num_classes],
        train_samples: options[:train_samples],
        val_samples: options[:val_samples],
        test_samples: options[:test_samples],
        seed: options[:seed]
      )
      val_data = Dataset.prepare_dataset(
        batch_size: options[:batch_size],
        split: "validation",
        data_file: options[:data_file],
        synthetic: options[:synthetic],
        input_res: options[:input_res],
        num_classes: options[:num_classes],
        train_samples: options[:train_samples],
        val_samples: options[:val_samples],
        test_samples: options[:test_samples],
        seed: options[:seed]
      )

      best_ckpt = options[:best_ckpt]
      if best_ckpt.nil? || best_ckpt.empty?
        best_ckpt = File.join(Dir.tmpdir, "speechcommands_best_#{SecureRandom.hex(8)}.npz")
      end
      best_acc = -Float::INFINITY
      best_epoch = 0

      options[:epochs].times do |epoch|
        tr_loss, tr_acc, tr_throughput = train_epoch(model, train_data, optimizer, epoch)
        MLX::Core.eval(tr_loss, tr_acc, tr_throughput)
        puts [
          "Epoch: #{epoch}",
          format("avg. Train loss %.3f", tr_loss.item.to_f),
          format("avg. Train acc %.3f", tr_acc.item.to_f),
          format("Throughput: %.2f samples/sec", tr_throughput.item.to_f)
        ].join(" | ")

        val_acc, val_throughput = test_epoch(model, val_data)
        MLX::Core.eval(val_acc, val_throughput)
        puts format(
          "Epoch: %d | Val acc %.3f | Throughput: %.2f samples/sec",
          epoch,
          val_acc.item.to_f,
          val_throughput.item.to_f
        )

        if val_acc.item.to_f >= best_acc
          best_acc = val_acc.item.to_f
          best_epoch = epoch
          model.save_weights(best_ckpt)
        end
      end

      puts "Testing best model from epoch #{best_epoch}"
      model.load_weights(best_ckpt) if File.exist?(best_ckpt)

      test_data = Dataset.prepare_dataset(
        batch_size: options[:batch_size],
        split: "test",
        data_file: options[:data_file],
        synthetic: options[:synthetic],
        input_res: options[:input_res],
        num_classes: options[:num_classes],
        train_samples: options[:train_samples],
        val_samples: options[:val_samples],
        test_samples: options[:test_samples],
        seed: options[:seed]
      )
      test_acc, _ = test_epoch(model, test_data)
      MLX::Core.eval(test_acc)
      puts format("Test acc -> %.3f", test_acc.item.to_f)

      model
    ensure
      if !best_ckpt.nil? && best_ckpt.include?("speechcommands_best_") && File.exist?(best_ckpt)
        File.delete(best_ckpt)
      end
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    arch: "kwt1",
    batch_size: 256,
    epochs: 100,
    lr: 1e-3,
    seed: 0,
    cpu: false,
    synthetic: true,
    data_file: nil,
    num_classes: 35,
    input_res: [98, 40],
    patch_res: [1, 40],
    dropout: 0.0,
    emb_dropout: 0.1,
    train_samples: 8_000,
    val_samples: 1_000,
    test_samples: 1_000,
    best_ckpt: nil
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby speechcommands/main.rb [options]"
    opts.on("--arch NAME", String, "Architecture: #{SpeechcommandsExample::Train::ARCHES.join(', ')}") { |v| options[:arch] = v }
    opts.on("--batch-size N", Integer, "Batch size") { |v| options[:batch_size] = v }
    opts.on("--epochs N", Integer, "Number of epochs") { |v| options[:epochs] = v }
    opts.on("--lr N", Float, "Learning rate") { |v| options[:lr] = v }
    opts.on("--seed N", Integer, "Random seed") { |v| options[:seed] = v }
    opts.on("--cpu", "Use CPU only") { options[:cpu] = true }
    opts.on("--synthetic", "Use synthetic MFSC-like data") { options[:synthetic] = true }
    opts.on("--real", "Use preprocessed real data from --data-file") { options[:synthetic] = false }
    opts.on("--data-file PATH", String, "NPZ file with split arrays: train_audio/train_label/...") { |v| options[:data_file] = v }
    opts.on("--num-classes N", Integer, "Number of keyword classes") { |v| options[:num_classes] = v }
    opts.on("--input-res H,W", Array, "Input resolution (time,freq)") { |v| options[:input_res] = v.map(&:to_i) }
    opts.on("--patch-res H,W", Array, "Patch resolution (time,freq)") { |v| options[:patch_res] = v.map(&:to_i) }
    opts.on("--dropout N", Float, "Transformer dropout") { |v| options[:dropout] = v }
    opts.on("--emb-dropout N", Float, "Embedding dropout") { |v| options[:emb_dropout] = v }
    opts.on("--train-samples N", Integer, "Synthetic train split size") { |v| options[:train_samples] = v }
    opts.on("--val-samples N", Integer, "Synthetic validation split size") { |v| options[:val_samples] = v }
    opts.on("--test-samples N", Integer, "Synthetic test split size") { |v| options[:test_samples] = v }
    opts.on("--best-ckpt PATH", String, "Path for best checkpoint weights") { |v| options[:best_ckpt] = v }
  end
  parser.parse!

  SpeechcommandsExample::Train.run(options)
end
