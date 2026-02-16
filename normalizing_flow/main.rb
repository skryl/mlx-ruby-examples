# frozen_string_literal: true

require "optparse"
require "time"

require_relative "flows"

module NormalizingFlowExample
  module Train
    module_function

    def gaussian_noise(rng)
      u1 = [rng.rand, 1e-12].max
      u2 = rng.rand
      Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
    end

    def get_moons_dataset(n_samples: 100_000, noise: 0.06, seed: 0)
      rng = Random.new(seed)
      n_first = n_samples / 2
      n_second = n_samples - n_first

      data = []
      n_first.times do
        theta = rng.rand * Math::PI
        x = Math.cos(theta)
        y = Math.sin(theta)
        data << [x + (noise * gaussian_noise(rng)), y + (noise * gaussian_noise(rng))]
      end
      n_second.times do
        theta = rng.rand * Math::PI
        x = 1.0 - Math.cos(theta)
        y = 0.5 - Math.sin(theta)
        data << [x + (noise * gaussian_noise(rng)), y + (noise * gaussian_noise(rng))]
      end

      mean0 = data.sum { |row| row[0] } / data.length.to_f
      mean1 = data.sum { |row| row[1] } / data.length.to_f
      var0 = data.sum { |row| (row[0] - mean0)**2 } / data.length.to_f
      var1 = data.sum { |row| (row[1] - mean1)**2 } / data.length.to_f
      std0 = Math.sqrt(var0)
      std1 = Math.sqrt(var1)

      normalized = data.map do |x0, x1|
        [(x0 - mean0) / std0, (x1 - mean1) / std1]
      end
      MLX::Core.array(normalized, MLX::Core.float32)
    end

    def loss_fn(model, x)
      MLX::Core.negative(MLX::Core.mean(model.call(x)))
    end

    def run(options)
      MLX::Core.set_default_device(MLX::Core.cpu) if options[:cpu]
      MLX::Core.random_seed(options[:seed])

      x = get_moons_dataset(
        n_samples: options[:n_samples],
        noise: options[:noise],
        seed: options[:seed]
      )

      model = RealNVP.new(
        options[:n_transforms],
        options[:d_params],
        options[:d_hidden],
        options[:n_layers]
      )
      MLX::Core.eval(model.parameters)
      optimizer = MLX::Optimizers::Adam.new(learning_rate: options[:learning_rate])
      trainer = model.trainer(optimizer: optimizer) do |x:|
        loss_fn(model, x)
      end

      train_data = lambda do |epoch:, **_kwargs|
        MLX::DSL::Data
          .from(0...x.shape[0])
          .shuffle(seed: options[:seed] + epoch.to_i + 1)
          .take(options[:n_batch])
          .batch(options[:n_batch], drop_last: true)
          .map do |batch_ids|
            ids = MLX::Core.array(batch_ids, MLX::Core.int32)
            MLX::Core.take(x, ids, 0)
          end
      end

      tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      trainer.after_epoch do |ctx|
        step = ctx.fetch(:epoch).to_i + 1
        next unless (step % options[:report_every]).zero?

        toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        puts format(
          "Step %d: Loss %.4f | It/sec %.2f",
          step,
          ctx.fetch(:epoch_loss).to_f,
          options[:report_every] / [toc - tic, 1e-9].max
        )
        tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      end
      trainer.register_dataflow(
        :flow_train,
        train: {
          collate: :x,
          reduce: :mean
        }
      )
      split_plan = MLX::DSL.splits do
        train(train_data)
      end

      trainer.fit_report(
        split_plan,
        **trainer.use_dataflow(:flow_train),
        epochs: options[:n_steps],
        keep_losses: false,
        strict_data_reuse: true
      )

      outputs = {}
      (0..options[:n_transforms]).each do |count|
        outputs[:"transform_#{count}"] = model.sample(
          [options[:n_plot_samples], options[:d_params]],
          n_transforms: count
        )
      end
      outputs[:original] = x
      MLX::Core.savez(options[:output], **outputs)
      puts "Saved sample arrays to #{options[:output]}"

      model
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    n_steps: 5_000,
    n_batch: 64,
    n_transforms: 6,
    d_params: 2,
    d_hidden: 128,
    n_layers: 4,
    learning_rate: 3e-4,
    noise: 0.06,
    cpu: false,
    seed: 0,
    n_samples: 100_000,
    n_plot_samples: 100_000,
    output: "normalizing_flow/samples.npz",
    report_every: 100
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby normalizing_flow/main.rb [options]"
    opts.on("--n-steps N", Integer, "Number of training steps") { |v| options[:n_steps] = v }
    opts.on("--n-batch N", Integer, "Batch size") { |v| options[:n_batch] = v }
    opts.on("--n-transforms N", Integer, "Number of flow transforms") { |v| options[:n_transforms] = v }
    opts.on("--d-params N", Integer, "Data dimensionality") { |v| options[:d_params] = v }
    opts.on("--d-hidden N", Integer, "Conditioner hidden dimension") { |v| options[:d_hidden] = v }
    opts.on("--n-layers N", Integer, "Conditioner MLP layers") { |v| options[:n_layers] = v }
    opts.on("--learning-rate N", Float, "Learning rate") { |v| options[:learning_rate] = v }
    opts.on("--noise N", Float, "Two moons noise level") { |v| options[:noise] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--n-samples N", Integer, "Two moons sample count") { |v| options[:n_samples] = v }
    opts.on("--n-plot-samples N", Integer, "Samples per saved transform") { |v| options[:n_plot_samples] = v }
    opts.on("--output PATH", String, "Output .npz path for generated samples") { |v| options[:output] = v }
    opts.on("--report-every N", Integer, "Report interval in training steps") { |v| options[:report_every] = v }
    opts.on("--cpu", "Use CPU device") { options[:cpu] = true }
  end
  parser.parse!

  NormalizingFlowExample::Train.run(options)
end
