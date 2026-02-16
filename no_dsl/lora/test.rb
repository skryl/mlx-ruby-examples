# frozen_string_literal: true

require "json"
require "optparse"
require "tmpdir"

require_relative "fuse"
require_relative "lora"
require_relative "../../benchmark/parity"

if $PROGRAM_NAME == __FILE__
  options = { seed: 53 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby lora/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  # LoRALinear forward/basic update sanity.
  linear = MLX::NN::Linear.new(8, 6, bias: false)
  lora = LoraExample::LoRALinear.from_linear(linear, rank: 2)
  x = MLX::Core.normal([4, 8])
  y = lora.call(x)
  MLX::Core.eval(y)
  raise "LoRALinear forward shape mismatch: #{y.shape.inspect}" unless y.shape == [4, 6]

  # Synthetic model load and LoRA injection sanity.
  model, tokenizer, = LoraExample::Utils.load("unused", { "add_eos_token" => true }, synthetic: true)
  raise "Tokenizer eos id missing" if tokenizer.eos_token_id.nil?
  LoraExample::Train.inject_lora(model, 1)
  unless model.model.layers.last.self_attn.q_proj.is_a?(LoraExample::LoRALinear)
    raise "LoRA injection failed for q_proj"
  end

  Dir.mktmpdir("lora-test-") do |dir|
    data_dir = File.join(dir, "data")
    Dir.mkdir(data_dir)

    train_lines = [
      { "text" => "select count from table where x equals y" },
      { "text" => "translate this sentence to sql" },
      { "text" => "group by user and compute average" },
      { "text" => "find all rows where score is greater than ten" }
    ]
    valid_lines = [
      { "text" => "select sum from transactions" },
      { "text" => "show rows for user alice" }
    ]
    test_lines = [
      { "text" => "return the max value per region" },
      { "text" => "count records with active status" }
    ]
    File.binwrite(File.join(data_dir, "train.jsonl"), train_lines.map { |x| JSON.generate(x) }.join("\n") + "\n")
    File.binwrite(File.join(data_dir, "valid.jsonl"), valid_lines.map { |x| JSON.generate(x) }.join("\n") + "\n")
    File.binwrite(File.join(data_dir, "test.jsonl"), test_lines.map { |x| JSON.generate(x) }.join("\n") + "\n")

    adapter_file = File.join(dir, "adapters.npz")
    model = LoraExample::Train.run(
      model: "unused",
      max_tokens: 8,
      temp: 0.0,
      prompt: "hello",
      train: true,
      add_eos_token: 1,
      data: data_dir,
      lora_layers: 1,
      batch_size: 2,
      iters: 3,
      val_batches: 1,
      learning_rate: 1e-3,
      steps_per_report: 1,
      steps_per_eval: 2,
      resume_adapter_file: nil,
      adapter_file: adapter_file,
      save_every: 2,
      test: true,
      test_batches: 1,
      seed: options[:seed],
      synthetic_model: true,
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    raise "Training did not return model" if model.nil?
    raise "Adapter file missing after training" unless File.exist?(adapter_file)

    # Fuse adapters back into base model in synthetic mode.
    save_path = File.join(dir, "fused")
    fuse_ok = system(
      "ruby",
      File.join(__dir__, "fuse.rb"),
      "--adapter-file", adapter_file,
      "--save-path", save_path,
      "--synthetic-model"
    )
    raise "Fuse command failed" unless fuse_ok
    raise "Fused model weights missing" unless File.exist?(File.join(save_path, "weights.npz"))
    raise "Fused model config missing" unless File.exist?(File.join(save_path, "config.json"))
  end

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "lora",
      inputs: { x: x },
      outputs: { y: y },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
  end

  puts "Tests pass :)"
end
