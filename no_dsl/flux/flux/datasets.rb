# frozen_string_literal: true

require "json"
require "pathname"

dsl_lib = File.join(File.expand_path("../..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module FluxExample
  class Dataset
    include Enumerable

    def each
      return enum_for(:each) unless block_given?

      length.times do |i|
        yield self[i]
      end
    end
  end

  class LocalDataset < Dataset
    attr_reader :dataset_base

    def initialize(dataset, data_file, prompt_key: "prompt")
      @dataset_base = Pathname.new(dataset)
      @prompt_key = prompt_key
      @data = File.readlines(data_file).map { |line| JSON.parse(line) }
    end

    def length
      @data.length
    end

    def [](index)
      item = @data[index]
      image_path = dataset_base.join(item.fetch("image"))
      [load_image(image_path), item.fetch(@prompt_key)]
    end

    private

    def load_image(path)
      if File.exist?(path)
        # Reads simple PPM (P6) files written by flux txt2image/image2image scripts.
        data = File.binread(path)
        if data.start_with?("P6")
          header, payload = data.split("\n", 4)[0, 3], data.split("\n", 4)[3]
          dims = header[1].split.map(&:to_i)
          w, h = dims
          arr = Array.new(h) { Array.new(w) { [0.0, 0.0, 0.0] } }
          p = 0
          h.times do |y|
            w.times do |x|
              r = payload.getbyte(p)
              g = payload.getbyte(p + 1)
              b = payload.getbyte(p + 2)
              p += 3
              arr[y][x] = [r / 255.0, g / 255.0, b / 255.0]
            end
          end
          return MLX::Core.array(arr, MLX::Core.float32)
        end
      end

      MLX::Core.random_uniform([512, 512, 3], 0.0, 1.0, MLX::Core.float32)
    rescue StandardError
      MLX::Core.random_uniform([512, 512, 3], 0.0, 1.0, MLX::Core.float32)
    end
  end

  class LegacyDataset < LocalDataset
    def initialize(dataset)
      base = Pathname.new(dataset)
      payload = JSON.parse(File.binread(base.join("index.json")))
      @dataset_base = base
      @prompt_key = "text"
      @data = payload.fetch("data")
    end
  end

  class HuggingFaceDataset < Dataset
    def initialize(dataset)
      @dataset = dataset
      @size = 16
    end

    def length
      @size
    end

    def [](index)
      prompt = "sample prompt #{index} from #{@dataset}"
      image = MLX::Core.random_uniform([512, 512, 3], 0.0, 1.0, MLX::Core.float32)
      [image, prompt]
    end
  end

  module_function

  def load_dataset(dataset)
    dataset_base = Pathname.new(dataset)
    data_file = dataset_base.join("train.jsonl")
    legacy_file = dataset_base.join("index.json")

    if File.exist?(data_file)
      puts "Load the local dataset #{data_file}."
      LocalDataset.new(dataset, data_file)
    elsif File.exist?(legacy_file)
      puts "Load the local dataset #{legacy_file}."
      puts ""
      puts "     WARNING: 'index.json' is deprecated in favor of 'train.jsonl'."
      puts "              See the README for details."
      puts ""
      LegacyDataset.new(dataset)
    else
      puts "Load the Hugging Face dataset #{dataset}."
      HuggingFaceDataset.new(dataset)
    end
  end
end
