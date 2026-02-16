# frozen_string_literal: true

require_relative "../mnist/mnist"

module CvaeExample
  module Dataset
    module_function

    DATASETS = %w[mnist fashion_mnist].freeze

    class BatchIterator
      include Enumerable

      def initialize(images, labels, batch_size:, shuffle:, seed:, img_size:)
        @images = images
        @labels = labels
        @batch_size = batch_size
        @shuffle = shuffle
        @rng = Random.new(seed)
        @img_height = img_size[0]
        @img_width = img_size[1]

        @order = (0...@labels.shape[0]).to_a
        @position = 0
        reset
      end

      def each
        return enum_for(:each) unless block_given?

        while @position < @order.length
          batch_ids = @order[@position, @batch_size]
          @position += @batch_size
          next if batch_ids.nil? || batch_ids.empty?

          ids = MLX::Core.array(batch_ids, MLX::Core.int32)
          batch_images = MLX::Core.take(@images, ids, 0)
          batch_labels = MLX::Core.take(@labels, ids, 0)
          batch_images = preprocess_images(batch_images)
          yield({ "image" => batch_images, "label" => batch_labels })
        end
      end

      def reset
        @position = 0
        @order = (0...@labels.shape[0]).to_a
        @order.shuffle!(random: @rng) if @shuffle
        self
      end

      def size
        (@labels.shape[0].to_f / @batch_size).ceil
      end

      private

      def preprocess_images(images)
        out = images.astype(MLX::Core.float32)

        out = if out.shape.length == 2
          MLX::Core.reshape(out, [out.shape[0], 28, 28, 1])
        elsif out.shape.length == 3
          MLX::Core.expand_dims(out, 3)
        else
          out
        end

        resize_nearest(out, @img_height, @img_width)
      end

      def resize_nearest(images, out_height, out_width)
        in_height = images.shape[1]
        in_width = images.shape[2]
        return images if in_height == out_height && in_width == out_width

        h_idx = MLX::Core.array((0...out_height).map { |i| (i * in_height) / out_height }, MLX::Core.int32)
        w_idx = MLX::Core.array((0...out_width).map { |i| (i * in_width) / out_width }, MLX::Core.int32)

        resized = MLX::Core.take(images, h_idx, 1)
        MLX::Core.take(resized, w_idx, 2)
      end
    end

    def mnist(
      batch_size:,
      img_size:,
      root: "/tmp",
      dataset: "mnist",
      synthetic: false,
      train_size: 60_000,
      test_size: 10_000,
      seed: 0,
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
      unless DATASETS.include?(dataset)
        raise ArgumentError, "dataset must be one of #{DATASETS.join(', ')}"
      end

      loader = dataset == "mnist" ? MnistExample::Dataset.method(:mnist) : MnistExample::Dataset.method(:fashion_mnist)
      train_images, train_labels, test_images, test_labels = loader.call(
        save_dir: root,
        python_bin: python_bin,
        synthetic: synthetic,
        train_size: train_size,
        test_size: test_size,
        seed: seed
      )

      train_iter = BatchIterator.new(
        train_images,
        train_labels,
        batch_size: batch_size,
        shuffle: true,
        seed: seed,
        img_size: img_size
      )
      test_iter = BatchIterator.new(
        test_images,
        test_labels,
        batch_size: batch_size,
        shuffle: false,
        seed: seed + 1,
        img_size: img_size
      )
      [train_iter, test_iter]
    end
  end
end
