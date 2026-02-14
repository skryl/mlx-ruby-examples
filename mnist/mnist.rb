# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module MnistExample
  module Dataset
    module_function

    SCRIPT_PATH = Pathname.new(__dir__).join("python", "prepare_dataset.py").to_s
    DEFAULT_BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    FASHION_BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    def mnist(
      save_dir: "/tmp",
      base_url: DEFAULT_BASE_URL,
      filename: "mnist.npz",
      python_bin: ENV.fetch("PYTHON_BIN", "python3"),
      synthetic: false,
      train_size: 60_000,
      test_size: 10_000,
      seed: 0
    )
      return synthetic_mnist(train_size: train_size, test_size: test_size, seed: seed) if synthetic

      save_path = Pathname.new(save_dir).expand_path.join(filename)
      ensure_prepared(save_path, base_url, python_bin)
      load_npz(save_path)
    end

    def fashion_mnist(
      save_dir: "/tmp",
      python_bin: ENV.fetch("PYTHON_BIN", "python3"),
      synthetic: false,
      train_size: 60_000,
      test_size: 10_000,
      seed: 0
    )
      mnist(
        save_dir: save_dir,
        base_url: FASHION_BASE_URL,
        filename: "fashion_mnist.npz",
        python_bin: python_bin,
        synthetic: synthetic,
        train_size: train_size,
        test_size: test_size,
        seed: seed
      )
    end

    def synthetic_mnist(train_size:, test_size:, seed:)
      rng = Random.new(seed)
      train_images = MLX::Core.random_uniform([train_size, 28 * 28], 0.0, 1.0, MLX::Core.float32)
      test_images = MLX::Core.random_uniform([test_size, 28 * 28], 0.0, 1.0, MLX::Core.float32)
      train_labels = MLX::Core.array(Array.new(train_size) { rng.rand(0...10) }, MLX::Core.int32)
      test_labels = MLX::Core.array(Array.new(test_size) { rng.rand(0...10) }, MLX::Core.int32)
      [train_images, train_labels, test_images, test_labels]
    end

    def ensure_prepared(save_path, base_url, python_bin)
      return if File.exist?(save_path)

      save_path.dirname.mkpath
      stdout, stderr, status = Open3.capture3(python_bin, SCRIPT_PATH, save_path.to_s, base_url.to_s)
      return if status.success? && File.exist?(save_path)

      raise RuntimeError, "Failed to prepare MNIST dataset: #{stderr}\n#{stdout}"
    end

    def load_npz(path)
      data = MLX::Core.load(path.to_s).to_a.to_h.transform_keys(&:to_s)
      train_x = data.fetch("training_images").astype(MLX::Core.float32)
      train_y = data.fetch("training_labels").astype(MLX::Core.int32)
      test_x = data.fetch("test_images").astype(MLX::Core.float32)
      test_y = data.fetch("test_labels").astype(MLX::Core.int32)
      [train_x, train_y, test_x, test_y]
    end
  end
end

if $PROGRAM_NAME == __FILE__
  train_x, train_y, test_x, test_y = MnistExample::Dataset.mnist
  raise "Wrong training set size" unless train_x.shape == [60_000, 28 * 28]
  raise "Wrong training labels size" unless train_y.shape == [60_000]
  raise "Wrong test set size" unless test_x.shape == [10_000, 28 * 28]
  raise "Wrong test labels size" unless test_y.shape == [10_000]
  puts "Tests pass :)"
end
