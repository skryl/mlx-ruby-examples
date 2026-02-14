# frozen_string_literal: true

require "optparse"
require "time"

require_relative "audio"
require_relative "decoding"
require_relative "load_models"
require_relative "transcribe"

module WhisperExample
  module Benchmark
    module_function

    TEST_AUDIO = File.join(__dir__, "assets", "ls_test.flac")

    def timer(iterations: 10)
      5.times { yield }
      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      iterations.times { yield }
      stop = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      (stop - start) / iterations
    end

    def feats(n_mels: 80)
      data = MLX::Core.random_uniform([Audio::N_SAMPLES], -1.0, 1.0, MLX::Core.float32)
      mel = Audio.log_mel_spectrogram(data, n_mels: n_mels)
      MLX::Core.eval(mel)
      mel
    end

    def model_forward(model, mels, tokens)
      logits = model.call(mels, tokens)
      MLX::Core.eval(logits)
      logits
    end

    def decode(model, mels)
      Decoding.decode(model, mels)
    end

    def everything(model_path)
      data = MLX::Core.random_uniform([Audio::N_SAMPLES], -1.0, 1.0, MLX::Core.float32)
      Transcribe.transcribe(data, path_or_hf_repo: model_path)
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { models: ["tiny"] }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby whisper/benchmark.rb [options]"
    opts.on("--models LIST", String, "Comma-separated model aliases") { |v| options[:models] = v.split(",") }
  end
  parser.parse!

  puts "Selected models: #{options[:models].join(', ')}"

  feat_time = WhisperExample::Benchmark.timer { WhisperExample::Benchmark.feats }
  puts format("Feature time %.3f s", feat_time)

  options[:models].each do |name|
    model_path = "mlx-community/whisper-#{name}"
    begin
      model = WhisperExample::LoadModels.load_model(model_path, dtype: MLX::Core.float16)
    rescue StandardError
      dims = WhisperExample::ModelDimensions.new
      model = WhisperExample::Whisper.new(dims, dtype: MLX::Core.float16)
    end

    mels = MLX::Core.expand_dims(WhisperExample::Benchmark.feats(n_mels: model.dims.n_mels), 0).astype(MLX::Core.float16)
    tokens = MLX::Core.array([[50_300, 100, 120, 130]], MLX::Core.int32)

    forward_t = WhisperExample::Benchmark.timer { WhisperExample::Benchmark.model_forward(model, mels, tokens) }
    decode_t = WhisperExample::Benchmark.timer { WhisperExample::Benchmark.decode(model, mels[0]) }

    puts "\nModel: #{name}"
    puts format("Forward time %.3f s", forward_t)
    puts format("Decode time %.3f s", decode_t)
  end
end
