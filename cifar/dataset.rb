# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module CifarExample
  module Dataset
    module_function

    PREPARE_SCRIPT_PATH = Pathname.new(__dir__).join("python", "prepare_cifar10.py").to_s

    class BatchIterator
      include Enumerable

      def initialize(images, labels, batch_size:, shuffle:, augment:, seed:, group:)
        @images = images
        @labels = labels
        @batch_size = batch_size
        @shuffle = shuffle
        @augment = augment
        @rng = Random.new(seed)
        @group = group
        @position = 0

        indices = (0...@labels.shape[0]).to_a
        if @group.size > 1
          indices = indices.select.with_index { |_idx, i| (i % @group.size) == @group.rank }
        end
        @indices = indices
        @order = @indices.dup
        @order.shuffle!(random: @rng) if @shuffle
      end

      def each
        return enum_for(:each) unless block_given?

        while @position < @order.length
          batch_ids = @order[@position, @batch_size]
          @position += @batch_size
          next if batch_ids.nil? || batch_ids.empty?

          index_array = MLX::Core.array(batch_ids, MLX::Core.int32)
          batch_images = MLX::Core.take(@images, index_array, 0)
          batch_labels = MLX::Core.take(@labels, index_array, 0)
          batch_images = augment_batch(batch_images) if @augment
          yield({ "image" => batch_images, "label" => batch_labels })
        end
      end

      def reset
        @position = 0
        @order = @indices.dup
        @order.shuffle!(random: @rng) if @shuffle
        self
      end

      def size
        (@indices.length.to_f / @batch_size).ceil
      end

      private

      def augment_batch(batch_images)
        rows = []
        (0...batch_images.shape[0]).each do |i|
          img = batch_images[i]
          if @rng.rand < 0.5
            w_indices = MLX::Core.array((0...img.shape[1]).to_a.reverse, MLX::Core.int32)
            img = MLX::Core.take(img, w_indices, 1)
          end
          img = MLX::Core.pad(img, [[4, 4], [4, 4], [0, 0]])
          top = @rng.rand(0..8)
          left = @rng.rand(0..8)
          h_indices = MLX::Core.array((top...(top + 32)).to_a, MLX::Core.int32)
          w_indices = MLX::Core.array((left...(left + 32)).to_a, MLX::Core.int32)
          img = MLX::Core.take(img, h_indices, 0)
          img = MLX::Core.take(img, w_indices, 1)
          rows << MLX::Core.expand_dims(img, 0)
        end
        MLX::Core.concatenate(rows, 0)
      end
    end

    def normalize(images)
      mean = MLX::Core.array([0.485, 0.456, 0.406], MLX::Core.float32)
      std = MLX::Core.array([0.229, 0.224, 0.225], MLX::Core.float32)
      mean = MLX::Core.reshape(mean, [1, 1, 1, 3])
      std = MLX::Core.reshape(std, [1, 1, 1, 3])
      x = MLX::Core.divide(images.astype(MLX::Core.float32), 255.0)
      MLX::Core.divide(MLX::Core.subtract(x, mean), std)
    end

    def load_npz_dataset(file_path)
      data = MLX::Core.load(file_path.to_s).to_a.to_h.transform_keys(&:to_s)
      images = data["images"] || data["image"] || data["x"]
      labels = data["labels"] || data["label"] || data["y"]
      if images.nil? || labels.nil?
        raise ArgumentError, "Expected keys images/labels in #{file_path}"
      end
      [normalize(images), labels.astype(MLX::Core.int32)]
    end

    def synthetic_dataset(samples, seed:)
      rng = Random.new(seed)
      image_count = samples.to_i
      labels = Array.new(image_count) { rng.rand(0...10) }
      images = MLX::Core.random_uniform([image_count, 32, 32, 3], 0.0, 255.0, MLX::Core.float32)
      [normalize(images), MLX::Core.array(labels, MLX::Core.int32)]
    end

    def prepare_with_python(root_dir, python_bin:)
      stdout, stderr, status = Open3.capture3(python_bin, PREPARE_SCRIPT_PATH, root_dir.to_s)
      return true if status.success?

      warn "[WARN] Failed to prepare CIFAR-10 via Python bridge: #{stderr}"
      warn "[WARN] Output: #{stdout}" unless stdout.strip.empty?
      false
    end

    def get_cifar10(
      batch_size,
      root: nil,
      synthetic: false,
      train_samples: 50_000,
      test_samples: 10_000,
      seed: 0,
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
      root_dir = root.nil? ? Pathname.new(__dir__).join("data") : Pathname.new(root.to_s)
      root_dir = root_dir.expand_path

      train_images = nil
      train_labels = nil
      test_images = nil
      test_labels = nil

      if !synthetic
        train_file = root_dir.join("cifar10_train.npz")
        test_file = root_dir.join("cifar10_test.npz")
        if !train_file.exist? || !test_file.exist?
          prepare_with_python(root_dir, python_bin: python_bin)
        end
        if train_file.exist? && test_file.exist?
          train_images, train_labels = load_npz_dataset(train_file)
          test_images, test_labels = load_npz_dataset(test_file)
        else
          warn "[WARN] CIFAR-10 files not found, using synthetic data."
          synthetic = true
        end
      end

      if synthetic
        train_images, train_labels = synthetic_dataset(train_samples, seed: seed)
        test_images, test_labels = synthetic_dataset(test_samples, seed: seed + 1)
      end

      group = MLX::Core.init
      train_iter = BatchIterator.new(
        train_images,
        train_labels,
        batch_size: batch_size,
        shuffle: true,
        augment: true,
        seed: seed,
        group: group
      )
      test_iter = BatchIterator.new(
        test_images,
        test_labels,
        batch_size: batch_size,
        shuffle: false,
        augment: false,
        seed: seed + 1,
        group: group
      )
      [train_iter, test_iter]
    end
  end
end
