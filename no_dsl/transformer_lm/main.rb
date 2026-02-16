# frozen_string_literal: true

require "optparse"
require "time"

require_relative "datasets"

module TransformerLmExample
  class TransformerLM < MLX::NN::Module
    def initialize(vocab_size:, num_layers:, dims:, num_heads:, checkpoint:)
      super()
      self.embedding = MLX::NN::Embedding.new(vocab_size, dims)
      self.pe = MLX::NN::SinusoidalPositionalEncoding.new(dims)
      self.transformer = MLX::NN::TransformerEncoder.new(
        num_layers,
        dims,
        num_heads,
        norm_first: true,
        checkpoint: checkpoint
      )
      self.out_proj = MLX::NN::Linear.new(dims, vocab_size)
    end

    def call(x)
      length = x.shape[1]
      mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(length)
      hidden = embedding.call(x)
      hidden = MLX::Core.add(hidden, pe.call(MLX::Core.arange(0, length, 1)))
      hidden = transformer.call(hidden, mask)
      out_proj.call(hidden)
    end
  end

  module Train
    module_function

    DATASETS = %w[enwik8 ptb wikitext2 wikitext103].freeze

    def synthetic_dataset(total_tokens:, vocab_size:, seed:)
      rng = Random.new(seed)
      make_tokens = lambda do |count|
        MLX::Core.array(Array.new(count) { rng.rand(0...vocab_size) }, MLX::Core.int32)
      end

      vocab = Array.new(vocab_size) { |i| ["tok_#{i}", i] }.to_h
      train = make_tokens.call(total_tokens)
      valid = make_tokens.call([total_tokens / 10, 2048].max)
      test = make_tokens.call([total_tokens / 10, 2048].max)
      [vocab, train, valid, test]
    end

    def to_samples(context_size, dataset)
      window_size = context_size + 1
      samples = dataset.size / window_size
      if samples <= 0
        raise ArgumentError, "Dataset is too short for context_size=#{context_size} (need at least #{window_size} tokens)"
      end

      trimmed = samples * window_size
      trimmed_dataset = MLX::Core.slice(dataset, [0], [trimmed])
      MLX::Core.reshape(trimmed_dataset, [samples, window_size])
    end

    def iterate_batches(batch_size, context_size, dataset, seed:)
      inputs = to_samples(context_size, dataset)
      rng = Random.new(seed)

      Enumerator.new do |enum|
        loop do
          order = (0...inputs.shape[0]).to_a
          order.shuffle!(random: rng)
          order.each_slice(batch_size) do |batch_ids|
            ids = MLX::Core.array(batch_ids, MLX::Core.int32)
            enum << MLX::Core.take(inputs, ids, 0)
          end
        end
      end
    end

    def loss_fn(model, inputs, reduction: "mean")
      seq_len = inputs.shape[1]
      x = MLX::Core.slice(inputs, [0, 0], [inputs.shape[0], seq_len - 1])
      y = MLX::Core.slice(inputs, [0, 1], [inputs.shape[0], seq_len])
      logits = model.call(x)
      MLX::NN::Losses.cross_entropy(logits, y, reduction: reduction)
    end

    def eval_fn(model, dataset, context_size:, batch_size:)
      inputs = to_samples(context_size, dataset)
      loss = 0.0

      s = 0
      while s < inputs.shape[0]
        n = [batch_size, inputs.shape[0] - s].min
        batch = MLX::Core.slice(inputs, [s, 0], [s + n, inputs.shape[1]])
        batch_loss = loss_fn(model, batch, reduction: "sum")
        MLX::Core.eval(batch_loss)
        loss += batch_loss.item.to_f
        s += batch_size
      end

      tokens = inputs.size - inputs.shape[0]
      loss / tokens.to_f
    end

    def run(options)
      if options[:synthetic]
        vocab, train, valid, test = synthetic_dataset(
          total_tokens: options[:synthetic_tokens],
          vocab_size: options[:synthetic_vocab_size],
          seed: options[:seed]
        )
      else
        vocab, train, valid, test = Datasets.load_dataset(
          options[:dataset],
          save_dir: options[:data_root]
        )
      end

      model = TransformerLM.new(
        vocab_size: vocab.length,
        num_layers: options[:num_blocks],
        dims: options[:dim],
        num_heads: options[:num_heads],
        checkpoint: options[:checkpoint]
      )
      MLX::Core.eval(model.parameters)

      nparams = MLX::Utils.tree_flatten(model.parameters).sum do |name, value|
        name.to_s.include?("embedding") ? 0 : value.size
      end
      puts format("Training a transformer with %.3f M parameters", nparams / (1024.0**2))

      optimizer = MLX::Optimizers::AdamW.new(
        learning_rate: options[:learning_rate],
        weight_decay: options[:weight_decay]
      )
      loss_and_grad_fn = MLX::NN.value_and_grad(model, ->(inputs) { loss_fn(model, inputs) })

      train_iterator = iterate_batches(
        options[:batch_size],
        options[:context_size],
        train,
        seed: options[:seed]
      )

      losses = []
      tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      options[:num_iters].times do |it|
        warmup = if options[:lr_warmup] <= 0
          1.0
        else
          [1.0, it.to_f / options[:lr_warmup].to_f].min
        end
        optimizer.learning_rate = warmup * options[:learning_rate]

        inputs = train_iterator.next
        loss, grads = loss_and_grad_fn.call(inputs)
        optimizer.update(model, grads)
        MLX::Core.eval(loss, model.parameters, optimizer.state)
        losses << loss.item.to_f

        if ((it + 1) % options[:steps_per_report]).zero?
          toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          train_loss = losses.sum / [losses.length, 1].max.to_f
          puts format(
            "Iter %d: Train loss %.3f, It/sec %.3f",
            it + 1,
            train_loss,
            options[:steps_per_report] / [toc - tic, 1e-9].max
          )
          losses = []
          tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        end

        next unless ((it + 1) % options[:steps_per_eval]).zero?

        eval_tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        val_loss = eval_fn(
          model,
          valid,
          context_size: options[:context_size],
          batch_size: options[:batch_size]
        )
        eval_toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        val_ppl = Math.exp(val_loss)
        puts format(
          "Iter %d: Val loss %.3f, Val ppl %.3f, Val took %.3fs",
          it + 1,
          val_loss,
          val_ppl,
          eval_toc - eval_tic
        )
      end

      if options[:eval_test]
        test_loss = eval_fn(
          model,
          test,
          context_size: options[:context_size],
          batch_size: options[:batch_size]
        )
        puts format("Test loss %.3f, Test ppl %.3f.", test_loss, Math.exp(test_loss))
      end

      model
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    gpu: false,
    seed: 42,
    dataset: "ptb",
    context_size: 1024,
    num_blocks: 12,
    dim: 1024,
    num_heads: 16,
    checkpoint: false,
    batch_size: 2,
    num_iters: 100_000,
    learning_rate: 3e-4,
    weight_decay: 1e-5,
    lr_warmup: 200,
    steps_per_report: 10,
    steps_per_eval: 1000,
    eval_test: false,
    data_root: "/tmp",
    synthetic: false,
    synthetic_tokens: 100_000,
    synthetic_vocab_size: 1024
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby transformer_lm/main.rb [options]"
    opts.on("--gpu", "Use Metal back-end") { options[:gpu] = true }
    opts.on("--seed N", Integer, "Seed for RNG") { |v| options[:seed] = v }
    opts.on("--dataset NAME", String, "Dataset: #{TransformerLmExample::Train::DATASETS.join(', ')}") { |v| options[:dataset] = v }
    opts.on("--context-size N", Integer, "Context size in tokens") { |v| options[:context_size] = v }
    opts.on("--num-blocks N", Integer, "Number of Transformer blocks") { |v| options[:num_blocks] = v }
    opts.on("--dim N", Integer, "Embedding/hidden dimension") { |v| options[:dim] = v }
    opts.on("--num-heads N", Integer, "Number of attention heads") { |v| options[:num_heads] = v }
    opts.on("--checkpoint", "Enable gradient checkpointing") { options[:checkpoint] = true }
    opts.on("--batch-size N", Integer, "Minibatch size") { |v| options[:batch_size] = v }
    opts.on("--num-iters N", Integer, "Training iterations") { |v| options[:num_iters] = v }
    opts.on("--learning-rate N", Float, "AdamW learning rate") { |v| options[:learning_rate] = v }
    opts.on("--weight-decay N", Float, "Weight decay") { |v| options[:weight_decay] = v }
    opts.on("--lr-warmup N", Integer, "LR linear warmup iterations") { |v| options[:lr_warmup] = v }
    opts.on("--steps-per-report N", Integer, "Train steps between reporting") { |v| options[:steps_per_report] = v }
    opts.on("--steps-per-eval N", Integer, "Train steps between validation evals") { |v| options[:steps_per_eval] = v }
    opts.on("--eval-test", "Evaluate test set after training") { options[:eval_test] = true }
    opts.on("--data-root PATH", String, "Dataset root/cache directory") { |v| options[:data_root] = v }
    opts.on("--synthetic", "Use synthetic token dataset") { options[:synthetic] = true }
    opts.on("--synthetic-tokens N", Integer, "Total synthetic training tokens") { |v| options[:synthetic_tokens] = v }
    opts.on("--synthetic-vocab-size N", Integer, "Synthetic vocabulary size") { |v| options[:synthetic_vocab_size] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])
  MLX::Core.set_default_device(MLX::Core.cpu) unless options[:gpu]
  TransformerLmExample::Train.run(options)
end
