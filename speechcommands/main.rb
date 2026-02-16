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
      created_temp_checkpoint = false
      if best_ckpt.nil? || best_ckpt.empty?
        best_ckpt = File.join(Dir.tmpdir, "speechcommands_best_#{SecureRandom.hex(8)}.npz")
        created_temp_checkpoint = true
      end

      trainer = model.trainer(optimizer: optimizer) do |audio:, label:|
        output = model.call(audio)
        MLX::Core.mean(MLX::NN::Losses.cross_entropy(output, label))
      end
      trainer.artifact_policy(
        checkpoint: {
          path: best_ckpt,
          strategy: :best
        },
        retention: { keep_last_n: 1 }
      )

      epoch_started_at = {}
      trainer.before_epoch do |ctx|
        epoch_started_at[ctx.fetch(:epoch)] = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      end
      trainer.after_epoch do |ctx|
        epoch = ctx.fetch(:epoch)
        val_acc, val_throughput = test_epoch(model, val_data)
        MLX::Core.eval(val_acc, val_throughput)
        puts [
          "Epoch: #{epoch}",
          format("Train loss %.3f", ctx.fetch(:epoch_loss).to_f),
          format("Val loss %.3f", ctx.fetch(:val_loss).to_f),
          format("Val acc %.3f", val_acc.item.to_f),
          format("Val throughput %.2f samples/sec", val_throughput.item.to_f),
          format(
            "Time %.3fs",
            Process.clock_gettime(Process::CLOCK_MONOTONIC) - epoch_started_at.fetch(epoch, Process.clock_gettime(Process::CLOCK_MONOTONIC))
          )
        ].join(" | ")
      end

      train_source = lambda do |epoch:, **_kwargs|
        _ = epoch
        train_data.reset
        train_data
      end
      val_source = lambda do |epoch:, **_kwargs|
        _ = epoch
        val_data.reset
        val_data
      end
      trainer.register_dataflow(
        :speech_cls,
        train: { reduce: :mean },
        validation: { reduce: :mean }
      )
      split_plan = MLX::DSL.splits do
        train(train_source)
        validation(val_source)
      end

      trainer.fit_report(
        split_plan,
        **trainer.use_dataflow(:speech_cls),
        epochs: options[:epochs],
        monitor: :val_loss,
        monitor_mode: :min,
        patience: options[:patience],
        min_delta: options.fetch(:min_delta, 0.0),
        keep_losses: false,
        strict_data_reuse: true
      )

      best_epoch = nil
      if File.exist?(best_ckpt)
        payload = model.load_checkpoint(best_ckpt, optimizer: optimizer)
        metadata = payload.is_a?(Hash) ? payload["metadata"] : nil
        best_epoch = metadata["epoch"] if metadata.is_a?(Hash)
      end
      puts "Testing best model from epoch #{best_epoch.nil? ? 'n/a' : best_epoch}"

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
      if created_temp_checkpoint && !best_ckpt.nil? && File.exist?(best_ckpt)
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
    best_ckpt: nil,
    patience: nil,
    min_delta: 0.0
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
    opts.on("--patience N", Integer, "Early stopping patience on validation loss") { |v| options[:patience] = v }
    opts.on("--min-delta N", Float, "Minimum validation-loss delta for improvement") { |v| options[:min_delta] = v }
  end
  parser.parse!

  SpeechcommandsExample::Train.run(options)
end
