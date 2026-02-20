# frozen_string_literal: true

require "json"
require "optparse"
require "pathname"
require "time"

require_relative "models"
require_relative "utils"

module LoraExample
  class Dataset
    def initialize(path, key: "text")
      @key = key
      if path.nil? || !Pathname.new(path).exist?
        @data = []
      else
        @data = File.readlines(path).map { |line| JSON.parse(line) }
      end
    end

    def [](idx)
      @data[idx].fetch(@key)
    end

    def length
      @data.length
    end
  end

  module Train
    module_function

    def load_datasets(data_dir)
      root = Pathname.new(data_dir.to_s)
      train = Dataset.new(root.join("train.jsonl"))
      valid = Dataset.new(root.join("valid.jsonl"))
      test = Dataset.new(root.join("test.jsonl"))
      [train, valid, test]
    end

    def take_time_indices(batch, start_idx, end_idx_exclusive)
      indices = MLX::Core.array((start_idx...end_idx_exclusive).to_a, MLX::Core.int32)
      MLX::Core.take(batch, indices, 1)
    end

    def iterate_batches(dataset, tokenizer, batch_size, train: false, seed: 0)
      rng = Random.new(seed)

      Enumerator.new do |enum|
        loop do
          indices = (0...dataset.length).to_a
          indices.shuffle!(random: rng) if train

          MLX::DSL::Data
            .from(indices)
            .batch(batch_size, drop_last: true)
            .each do |index_batch|
            batch_tokens = []
            index_batch.each do |idx|
              text = dataset[idx]
              batch_tokens << tokenizer.encode(text.to_s)
            end
            lengths = batch_tokens.map(&:length)
            max_len = lengths.max
            next if max_len.nil? || max_len < 2

            batch_arr = Array.new(batch_size) { Array.new(max_len, 0) }
            batch_size.times do |j|
              tokens = batch_tokens[j]
              tokens.each_with_index do |tok, k|
                batch_arr[j][k] = tok
              end
            end

            batch = MLX::Core.array(batch_arr, MLX::Core.int32)
            inputs = take_time_indices(batch, 0, max_len - 1)
            targets = take_time_indices(batch, 1, max_len)
            enum << [inputs, targets, MLX::Core.array(lengths, MLX::Core.int32)]
          end

          break unless train
        end
      end
    end

    def loss_fn(model, inputs, targets, lengths)
      logits, = model.call(inputs)
      logits = logits.astype(MLX::Core.float32)

      positions = MLX::Core.expand_dims(MLX::Core.arange(0, inputs.shape[1], 1), 0)
      valid_lengths = MLX::Core.expand_dims(MLX::Core.subtract(lengths, 1), 1)
      length_mask = MLX::Core.less(positions, valid_lengths).astype(MLX::Core.float32)

      ce = MLX::NN::Losses.cross_entropy(logits, targets)
      ce = MLX::Core.multiply(ce, length_mask)
      ntoks = MLX::Core.sum(length_mask)
      loss = MLX::Core.divide(MLX::Core.sum(ce), ntoks)
      [loss, ntoks]
    end

    def evaluate(model, dataset, tokenizer, batch_size, num_batches: -1, seed: 0)
      all_losses = 0.0
      ntokens = 0.0
      count = 0

      iterate_batches(dataset, tokenizer, batch_size, train: false, seed: seed).each do |inputs, targets, lengths|
        loss, toks = loss_fn(model, inputs, targets, lengths)
        MLX::Core.eval(loss, toks)
        all_losses += (loss.item.to_f * toks.item.to_f)
        ntokens += toks.item.to_f
        count += 1
        break if num_batches != -1 && count >= num_batches
      end

      return Float::INFINITY if ntokens <= 0.0

      all_losses / ntokens
    end

    def save_trainable_adapters(path, model)
      payload = {}
      MLX::Utils.tree_flatten(model.trainable_parameters).each do |name, value|
        payload[name.to_s.to_sym] = value
      end
      MLX::Core.savez(path.to_s, **payload)
    end

    def inject_lora(model, lora_layers)
      model.freeze
      start = [model.model.layers.length - lora_layers, 0].max
      model.model.layers[start..].each do |layer|
        layer.self_attn.q_proj = LoRALinear.from_linear(layer.self_attn.q_proj)
        layer.self_attn.v_proj = LoRALinear.from_linear(layer.self_attn.v_proj)
      end
      model
    end

    def train(model, train_set, valid_set, optimizer, tokenizer, options)
      loss_value_and_grad = MLX::NN.value_and_grad(
        model,
        lambda do |inputs, targets, lengths|
          loss_fn(model, inputs, targets, lengths)
        end
      )

      losses = []
      n_tokens = 0.0
      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      batches = iterate_batches(
        train_set,
        tokenizer,
        options[:batch_size],
        train: true,
        seed: options[:seed]
      )

      options[:iters].times do |iter|
        inputs, targets, lengths = batches.next
        (lvalue, toks), grad = loss_value_and_grad.call(inputs, targets, lengths)
        optimizer.update(model, grad)
        MLX::Core.eval(model.parameters, optimizer.state, lvalue, toks)

        losses << lvalue.item.to_f
        n_tokens += toks.item.to_f

        if ((iter + 1) % options[:steps_per_report]).zero?
          stop = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          train_loss = losses.sum / [losses.length, 1].max.to_f
          puts format(
            "Iter %d: Train loss %.3f, It/sec %.3f, Tokens/sec %.3f",
            iter + 1,
            train_loss,
            options[:steps_per_report] / [stop - start, 1e-9].max,
            n_tokens / [stop - start, 1e-9].max
          )
          losses = []
          n_tokens = 0.0
          start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        end

        if iter.zero? || ((iter + 1) % options[:steps_per_eval]).zero?
          eval_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          val_loss = evaluate(
            model,
            valid_set,
            tokenizer,
            options[:batch_size],
            num_batches: options[:val_batches],
            seed: options[:seed] + 1
          )
          puts format(
            "Iter %d: Val loss %.3f, Val took %.3fs",
            iter + 1,
            val_loss,
            Process.clock_gettime(Process::CLOCK_MONOTONIC) - eval_start
          )
          start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        end

        if ((iter + 1) % options[:save_every]).zero?
          save_trainable_adapters(options[:adapter_file], model)
          puts "Iter #{iter + 1}: Saved adapter weights to #{options[:adapter_file]}."
        end
      end
    end

    def generate_text(model, prompt, tokenizer, options)
      $stdout.print(prompt)
      $stdout.flush

      prompt_ids = MLX::Core.array(tokenizer.encode(prompt), MLX::Core.int32)
      tokens = []
      skip = 0
      Utils.generate(prompt_ids, model, temp: options[:temp]).each_with_index do |token, n|
        token_id = token.item.to_i
        break if token_id == tokenizer.eos_token_id || n >= options[:max_tokens]

        tokens << token_id
        decoded = tokenizer.decode(tokens)
        if decoded.length - skip > 1
          $stdout.print(decoded[skip...-1])
          $stdout.flush
          skip = decoded.length - 1
        end
      end
      puts tokenizer.decode(tokens)[skip..].to_s
      puts "=" * 10
      puts "No tokens generated for this prompt" if tokens.empty?
    end

    def run(options)
      srand(options[:seed])
      MLX::Core.random_seed(options[:seed])

      tokenizer_cfg = { "add_eos_token" => !!options[:add_eos_token] }
      model, tokenizer, _config = Utils.load(
        options[:model],
        tokenizer_cfg,
        synthetic: options[:synthetic_model],
        python_bin: options[:python_bin]
      )

      inject_lora(model, options[:lora_layers])

      total_params = MLX::Utils.tree_flatten(model.parameters).sum { |_k, v| v.size } / 1_000_000.0
      trainable_params = MLX::Utils.tree_flatten(model.trainable_parameters).sum { |_k, v| v.size } / 1_000_000.0
      puts format("Total parameters %.3fM", total_params)
      puts format("Trainable parameters %.3fM", trainable_params)

      train_set, valid_set, test_set = load_datasets(options[:data])
      if options[:train] && train_set.length.zero?
        raise ArgumentError, "Training set missing or empty: #{options[:data]}/train.jsonl"
      end
      if options[:train] && valid_set.length.zero?
        raise ArgumentError, "Validation set missing or empty: #{options[:data]}/valid.jsonl"
      end
      if options[:test] && test_set.length.zero?
        raise ArgumentError, "Test set missing or empty: #{options[:data]}/test.jsonl"
      end

      if !options[:resume_adapter_file].nil?
        puts "Loading pretrained adapters from #{options[:resume_adapter_file]}"
        model.load_weights(options[:resume_adapter_file], strict: false)
      end

      if options[:train]
        puts "Training"
        optimizer = MLX::Optimizers::Adam.new(learning_rate: options[:learning_rate])
        train(model, train_set, valid_set, optimizer, tokenizer, options)
        save_trainable_adapters(options[:adapter_file], model)
      end

      if File.file?(options[:adapter_file])
        model.load_weights(options[:adapter_file], strict: false)
      elsif options[:train]
        raise "Adapter file missing after training: #{options[:adapter_file]}"
      end

      if options[:test]
        puts "Testing"
        model.eval
        test_loss = evaluate(
          model,
          test_set,
          tokenizer,
          options[:batch_size],
          num_batches: options[:test_batches],
          seed: options[:seed] + 2
        )
        puts format("Test loss %.3f, Test ppl %.3f.", test_loss, Math.exp(test_loss))
      end

      generate_text(model, options[:prompt], tokenizer, options) unless options[:prompt].nil?
      model
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model: "mlx_model",
    max_tokens: 100,
    temp: 0.8,
    prompt: nil,
    train: false,
    add_eos_token: 1,
    data: "lora/data",
    lora_layers: 16,
    batch_size: 4,
    iters: 1000,
    val_batches: 25,
    learning_rate: 1e-5,
    steps_per_report: 10,
    steps_per_eval: 200,
    resume_adapter_file: nil,
    adapter_file: "adapters.npz",
    save_every: 100,
    test: false,
    test_batches: 500,
    seed: 0,
    synthetic_model: true,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby lora/lora.rb [options]"
    opts.on("--model PATH", String, "Local model directory") { |v| options[:model] = v }
    opts.on("--max-tokens N", Integer, "Maximum generation tokens") { |v| options[:max_tokens] = v }
    opts.on("--temp N", Float, "Sampling temperature") { |v| options[:temp] = v }
    opts.on("--prompt TEXT", String, "Prompt for generation") { |v| options[:prompt] = v }
    opts.on("--train", "Enable training") { options[:train] = true }
    opts.on("--add-eos-token N", Integer, "Tokenizer add_eos_token flag (0/1)") { |v| options[:add_eos_token] = v }
    opts.on("--data DIR", String, "Directory with train/valid/test jsonl") { |v| options[:data] = v }
    opts.on("--lora-layers N", Integer, "Number of final layers with LoRA adapters") { |v| options[:lora_layers] = v }
    opts.on("--batch-size N", Integer, "Minibatch size") { |v| options[:batch_size] = v }
    opts.on("--iters N", Integer, "Training iterations") { |v| options[:iters] = v }
    opts.on("--val-batches N", Integer, "Validation batch count (-1 = all)") { |v| options[:val_batches] = v }
    opts.on("--learning-rate N", Float, "Adam learning rate") { |v| options[:learning_rate] = v }
    opts.on("--steps-per-report N", Integer, "Train steps between reporting") { |v| options[:steps_per_report] = v }
    opts.on("--steps-per-eval N", Integer, "Train steps between eval") { |v| options[:steps_per_eval] = v }
    opts.on("--resume-adapter-file PATH", String, "Adapter file to resume from") { |v| options[:resume_adapter_file] = v }
    opts.on("--adapter-file PATH", String, "Path to save/load LoRA adapters") { |v| options[:adapter_file] = v }
    opts.on("--save-every N", Integer, "Save adapters every N steps") { |v| options[:save_every] = v }
    opts.on("--test", "Evaluate on test set") { options[:test] = true }
    opts.on("--test-batches N", Integer, "Test batch count (-1 = all)") { |v| options[:test_batches] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--synthetic-model", "Use synthetic small base model (default)") { options[:synthetic_model] = true }
    opts.on("--real-model", "Load a real local model from --model") { options[:synthetic_model] = false }
    opts.on("--python-bin BIN", String, "Python binary for HF tokenizer bridge") { |v| options[:python_bin] = v }
  end
  parser.parse!

  LoraExample::Train.run(options)
end
