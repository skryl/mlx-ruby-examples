# frozen_string_literal: true

require "optparse"
require "pathname"
require "time"

require_relative "dataset"
require_relative "vae"

module CvaeExample
  module Train
    module_function

    def clamp_u8(value)
      pixel = (value.to_f * 255.0).round
      return 0 if pixel < 0
      return 255 if pixel > 255

      pixel
    end

    def save_grid_pgm(images, num_rows:, out_file:)
      image_batch = images.is_a?(Array) ? images : images.to_a
      raise ArgumentError, "Cannot save an empty image batch" if image_batch.empty?

      batch = image_batch.length
      num_rows = [[num_rows, 1].max, batch].min
      num_cols = (batch.to_f / num_rows).ceil

      height = image_batch[0].length
      width = image_batch[0][0].length
      grid_height = num_rows * height
      grid_width = num_cols * width
      pixels = Array.new(grid_height * grid_width, 0)

      image_batch.each_with_index do |image, idx|
        row = idx / num_cols
        col = idx % num_cols

        height.times do |i|
          width.times do |j|
            value = image[i][j]
            value = value[0] if value.is_a?(Array)
            grid_i = row * height + i
            grid_j = col * width + j
            pixels[(grid_i * grid_width) + grid_j] = clamp_u8(value)
          end
        end
      end

      out_path = Pathname.new(out_file.to_s)
      out_path.dirname.mkpath
      header = "P5\n#{grid_width} #{grid_height}\n255\n"
      File.binwrite(out_path, header + pixels.pack("C*"))
    end

    def loss_fn(model, x)
      x_recon, mu, logvar = model.call(x)
      recon_loss = MLX::NN::Losses.mse_loss(x_recon, x, reduction: "sum")

      mu_sq = MLX::Core.square(mu)
      logvar_exp = MLX::Core.exp(logvar)
      one_plus_logvar = MLX::Core.add(1.0, logvar)
      kl_inner = MLX::Core.subtract(one_plus_logvar, MLX::Core.add(mu_sq, logvar_exp))
      kl_div = MLX::Core.multiply(-0.5, MLX::Core.sum(kl_inner))

      MLX::Core.add(recon_loss, kl_div)
    end

    def reconstruct(model, batch, out_file)
      images = batch.fetch("image")
      images_recon, = model.call(images)
      MLX::Core.eval(images_recon)

      source = images.to_a
      recon = images_recon.to_a
      paired = []
      source.each_with_index do |img, i|
        paired << img
        paired << recon[i]
      end

      save_grid_pgm(paired, num_rows: 16, out_file: out_file)
    end

    def generate(model, out_file, num_samples: 128)
      z = MLX::Core.normal([num_samples, model.num_latent_dims])
      images = model.decode(z)
      MLX::Core.eval(images)
      num_rows = Math.sqrt(num_samples).floor
      num_rows = 1 if num_rows < 1
      save_grid_pgm(images.to_a, num_rows: num_rows, out_file: out_file)
    end

    def train_epoch(model, train_iter, loss_and_grad_fn, optimizer, epoch:)
      loss_acc = 0.0
      throughput_acc = 0.0
      batch_count = 0

      train_iter.each_with_index do |batch, idx|
        x = batch.fetch("image")
        throughput_tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)

        loss, grads = loss_and_grad_fn.call(x)
        optimizer.update(model, grads)
        MLX::Core.eval(loss, model.parameters, optimizer.state)

        throughput_toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        throughput_acc += x.shape[0] / [throughput_toc - throughput_tic, 1e-9].max
        loss_acc += loss.item.to_f
        batch_count += 1

        next unless idx.positive? && (idx % 10).zero?

        puts format(
          "Epoch %4d | Loss %10.2f | Throughput %8.2f im/s | Batch %5d",
          epoch,
          loss_acc / batch_count,
          throughput_acc / batch_count,
          idx
        )
      end

      denom = [batch_count, 1].max.to_f
      [loss_acc / denom, throughput_acc / denom]
    end

    def run(options)
      MLX::Core.set_default_device(MLX::Core.cpu) if options[:cpu]
      MLX::Core.random_seed(options[:seed])

      puts "Options:"
      puts "  Device: #{options[:cpu] ? 'CPU' : 'GPU'}"
      puts "  Seed: #{options[:seed]}"
      puts "  Batch size: #{options[:batch_size]}"
      puts "  Max filters: #{options[:max_filters]}"
      puts "  Epochs: #{options[:epochs]}"
      puts "  Learning rate: #{options[:lr]}"
      puts "  Latent dimensions: #{options[:latent_dims]}"

      img_shape = [options[:img_size], options[:img_size], 1]
      train_iter, test_iter = Dataset.mnist(
        batch_size: options[:batch_size],
        img_size: img_shape[0...2],
        root: options[:data_root],
        dataset: options[:dataset],
        synthetic: options[:synthetic],
        train_size: options[:train_size],
        test_size: options[:test_size],
        seed: options[:seed],
        python_bin: options[:python_bin]
      )

      save_dir = Pathname.new(options[:save_dir]).expand_path
      save_dir.mkpath

      model = CVAE.new(options[:latent_dims], img_shape, options[:max_filters])
      MLX::Core.eval(model.parameters)

      num_params = MLX::Utils.tree_flatten(model.trainable_parameters).sum { |_k, x| x.size }
      puts format("Number of trainable params: %.4f M", num_params / 1e6)

      optimizer = MLX::Optimizers::AdamW.new(learning_rate: options[:lr])
      loss_and_grad_fn = MLX::NN.value_and_grad(model, ->(x) { loss_fn(model, x) })

      train_batch = train_iter.first
      test_batch = test_iter.first

      (1..options[:epochs]).each do |epoch|
        train_iter.reset
        model.train(true)
        tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        avg_loss, avg_throughput = train_epoch(model, train_iter, loss_and_grad_fn, optimizer, epoch: epoch)
        toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)

        puts format(
          "Epoch %4d | Loss %10.2f | Throughput %8.2f im/s | Time %8.2f (s)",
          epoch,
          avg_loss,
          avg_throughput,
          toc - tic
        )

        model.eval
        if options[:save_images]
          reconstruct(model, train_batch, save_dir.join(format("train_%03d.pgm", epoch)))
          reconstruct(model, test_batch, save_dir.join(format("test_%03d.pgm", epoch)))
          generate(model, save_dir.join(format("generated_%03d.pgm", epoch)))
        end
        model.save_weights(save_dir.join("weights.npz").to_s)
      end

      model
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    cpu: false,
    seed: 0,
    batch_size: 128,
    max_filters: 64,
    epochs: 50,
    lr: 1e-3,
    latent_dims: 8,
    save_dir: "cvae/models",
    dataset: "mnist",
    data_root: "/tmp",
    python_bin: ENV.fetch("PYTHON_BIN", "python3"),
    synthetic: false,
    train_size: 60_000,
    test_size: 10_000,
    img_size: 64,
    save_images: true
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby cvae/main.rb [options]"
    opts.on("--cpu", "Use CPU instead of GPU acceleration") { options[:cpu] = true }
    opts.on("--seed N", Integer, "Random seed") { |v| options[:seed] = v }
    opts.on("--batch-size N", Integer, "Batch size for training") { |v| options[:batch_size] = v }
    opts.on("--max-filters N", Integer, "Maximum number of convolution filters") { |v| options[:max_filters] = v }
    opts.on("--epochs N", Integer, "Number of training epochs") { |v| options[:epochs] = v }
    opts.on("--lr N", Float, "Learning rate") { |v| options[:lr] = v }
    opts.on("--latent-dims N", Integer, "Number of latent dimensions") { |v| options[:latent_dims] = v }
    opts.on("--save-dir PATH", String, "Output directory for weights/images") { |v| options[:save_dir] = v }
    opts.on("--dataset NAME", String, "Dataset: #{CvaeExample::Dataset::DATASETS.join(', ')}") { |v| options[:dataset] = v }
    opts.on("--data-root PATH", String, "Dataset cache directory") { |v| options[:data_root] = v }
    opts.on("--python-bin BIN", String, "Python binary for dataset bridge") { |v| options[:python_bin] = v }
    opts.on("--synthetic", "Use synthetic MNIST data") { options[:synthetic] = true }
    opts.on("--train-size N", Integer, "Synthetic train size") { |v| options[:train_size] = v }
    opts.on("--test-size N", Integer, "Synthetic test size") { |v| options[:test_size] = v }
    opts.on("--img-size N", Integer, "Resized image size (HxW)") { |v| options[:img_size] = v }
    opts.on("--save-images", "Save reconstruction/sample images") { options[:save_images] = true }
    opts.on("--no-save-images", "Disable image export") { options[:save_images] = false }
  end
  parser.parse!

  CvaeExample::Train.run(options)
end
