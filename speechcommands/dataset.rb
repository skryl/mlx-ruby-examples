# frozen_string_literal: true

require "pathname"


require "mlx"

module SpeechcommandsExample
  module Dataset
    module_function

    class BatchIterator
      include Enumerable

      def initialize(audio, labels, batch_size:, shuffle:, seed:)
        @audio = audio
        @labels = labels
        @batch_size = batch_size
        @shuffle = shuffle
        @rng = Random.new(seed)
        @position = 0
        @order = (0...@labels.shape[0]).to_a
        @order.shuffle!(random: @rng) if @shuffle
      end

      def each
        return enum_for(:each) unless block_given?

        while @position < @order.length
          batch_ids = @order[@position, @batch_size]
          @position += @batch_size
          next if batch_ids.nil? || batch_ids.empty?

          ids = MLX::Core.array(batch_ids, MLX::Core.int32)
          batch_audio = MLX::Core.take(@audio, ids, 0)
          batch_label = MLX::Core.take(@labels, ids, 0)
          yield({ "audio" => batch_audio, "label" => batch_label })
        end
      end

      def reset
        @position = 0
        @order = (0...@labels.shape[0]).to_a
        @order.shuffle!(random: @rng) if @shuffle
        self
      end
    end

    def load_split_from_npz(data_file, split)
      data = MLX::Core.load(data_file.to_s).to_a.to_h.transform_keys(&:to_s)
      audio = data["#{split}_audio"] || data["#{split}.audio"] || data["audio"]
      labels = data["#{split}_label"] || data["#{split}.label"] || data["label"]
      if audio.nil? || labels.nil?
        raise ArgumentError, "Could not find #{split} audio/label arrays in #{data_file}"
      end
      [audio.astype(MLX::Core.float32), labels.astype(MLX::Core.int32)]
    end

    def synthetic_split(
      size:,
      num_classes:,
      input_res:,
      seed:,
      noise_scale: 0.25
    )
      labels = Array.new(size) { rand(num_classes) }
      label_arr = MLX::Core.array(labels, MLX::Core.int32)

      prototypes = MLX::Core.normal([num_classes, input_res[0], input_res[1], 1])
      audio = MLX::Core.take(prototypes, label_arr, 0)
      noise = MLX::Core.multiply(
        MLX::Core.normal([size, input_res[0], input_res[1], 1]),
        noise_scale
      )
      audio = MLX::Core.add(audio, noise).astype(MLX::Core.float32)

      mean = MLX::Core.mean(audio)
      std = MLX::Core.std(audio)
      audio = MLX::Core.divide(
        MLX::Core.subtract(audio, mean),
        MLX::Core.add(std, 1e-6)
      )
      [audio, label_arr]
    end

    def prepare_dataset(
      batch_size:,
      split:,
      data_file: nil,
      synthetic: true,
      input_res: [98, 40],
      num_classes: 35,
      train_samples: 8_000,
      val_samples: 1_000,
      test_samples: 1_000,
      seed: 0
    )
      split_seed = case split
      when "train" then seed
      when "validation" then seed + 1
      when "test" then seed + 2
      else
        raise ArgumentError, "Unsupported split=#{split.inspect}"
      end
      Kernel.srand(split_seed)

      audio, labels = if !synthetic
        raise ArgumentError, "Provide --data-file when --real is selected" if data_file.nil?

        load_split_from_npz(data_file, split)
      else
        size = case split
        when "train" then train_samples
        when "validation" then val_samples
        when "test" then test_samples
        end
        synthetic_split(
          size: size,
          num_classes: num_classes,
          input_res: input_res,
          seed: split_seed
        )
      end

      BatchIterator.new(
        audio,
        labels,
        batch_size: batch_size,
        shuffle: split == "train",
        seed: split_seed
      )
    end
  end
end
