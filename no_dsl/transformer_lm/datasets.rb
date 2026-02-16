# frozen_string_literal: true

require "fileutils"
require "open-uri"
require "open3"
require "pathname"

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module TransformerLmExample
  module Datasets
    module_function

    def load_dataset(dataname, save_dir: "/tmp")
      case dataname
      when "enwik8"
        enwik8(save_dir: save_dir)
      when "ptb"
        ptb(save_dir: save_dir)
      when "wikitext2"
        wikitext(dataset: "2", save_dir: save_dir)
      else
        wikitext(dataset: "103", save_dir: save_dir)
      end
    end

    def _load(save_dir, filenames)
      eos = "<eos>"
      vocab = {}

      train_path = File.join(save_dir, filenames[0])
      File.foreach(train_path) do |line|
        line.strip.split(" ").each do |token|
          next if token.empty?

          vocab[token] = vocab.length unless vocab.key?(token)
        end
      end
      vocab[eos] = vocab.length unless vocab.key?(eos)

      datasets = filenames.map do |name|
        ids = []
        File.foreach(File.join(save_dir, name)) do |line|
          words = line.strip.split(" ").reject(&:empty?)
          words << eos
          words.each { |w| ids << vocab.fetch(w) }
        end
        MLX::Core.array(ids, MLX::Core.int32)
      end

      [vocab, *datasets]
    end

    def wikitext(dataset: "2", save_dir: "/tmp")
      unless %w[2 103].include?(dataset)
        raise ArgumentError, "Dataset must be either \"2\" or \"103\", got #{dataset.inspect}"
      end

      filenames = ["wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens"]
      dataname = "wikitext-#{dataset}"
      save_root = Pathname.new(save_dir).expand_path
      data_dir = save_root.join(dataname)

      unless data_dir.exist?
        zip_name = "#{dataname}-v1.zip"
        zip_path = save_root.join(zip_name)
        download_file("https://s3.amazonaws.com/research.metamind.io/wikitext/#{zip_name}", zip_path)
        unzip(zip_path, save_root)
      end

      _load(data_dir.to_s, filenames)
    end

    def ptb(save_dir: "/tmp")
      filenames = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]
      save_root = Pathname.new(save_dir).expand_path.join("ptb")
      save_root.mkpath unless save_root.exist?

      base_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
      filenames.each do |name|
        out_file = save_root.join(name)
        download_file(base_url + name, out_file) unless out_file.exist?
      end

      _load(save_root.to_s, filenames)
    end

    def enwik8(save_dir: "/tmp")
      save_root = Pathname.new(save_dir).expand_path
      zip_path = save_root.join("enwik8.zip")
      download_file("http://mattmahoney.net/dc/enwik8.zip", zip_path) unless zip_path.exist?

      raw_path = save_root.join("enwik8")
      unless raw_path.exist?
        stdout, stderr, status = Open3.capture3("unzip", "-o", "-q", zip_path.to_s, "enwik8", "-d", save_root.to_s)
        unless status.success?
          raise RuntimeError, "Failed to unzip enwik8: #{stderr}\n#{stdout}"
        end
      end

      bytes = File.binread(raw_path).bytes
      num_test_bytes = 5_000_000
      train_bytes = bytes[0...(-2 * num_test_bytes)]
      valid_bytes = bytes[(-2 * num_test_bytes)...(-num_test_bytes)]
      test_bytes = bytes[-num_test_bytes..]

      vocab = {}
      train_bytes.each { |byte| vocab[byte] = vocab.length unless vocab.key?(byte) }

      to_array = lambda do |dataset|
        ids = dataset.map { |byte| vocab.fetch(byte) }
        MLX::Core.array(ids, MLX::Core.int32)
      end

      [vocab, to_array.call(train_bytes), to_array.call(valid_bytes), to_array.call(test_bytes)]
    end

    def download_file(url, out_file)
      out_path = Pathname.new(out_file.to_s)
      out_path.dirname.mkpath unless out_path.dirname.exist?
      URI.open(url) do |remote|
        File.binwrite(out_path, remote.read)
      end
    end

    def unzip(zip_file, destination)
      stdout, stderr, status = Open3.capture3(
        "unzip",
        "-o",
        "-q",
        zip_file.to_s,
        "-d",
        destination.to_s
      )
      return if status.success?

      raise RuntimeError, "Failed to unzip #{zip_file}: #{stderr}\n#{stdout}"
    end
  end
end

if $PROGRAM_NAME == __FILE__
  vocab, = TransformerLmExample::Datasets.enwik8
  raise "enwik8: Wrong vocab size" unless vocab.length == 205

  vocab, = TransformerLmExample::Datasets.ptb
  raise "PTB: Wrong vocab size" unless vocab.length == 10_000

  vocab, = TransformerLmExample::Datasets.wikitext
  raise "WikiText: Wrong vocab size" unless vocab.length == 33_279

  puts "Tests pass :)"
end
