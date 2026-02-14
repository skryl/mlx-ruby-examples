# frozen_string_literal: true

require "optparse"
require "tmpdir"

require_relative "encodec"
require_relative "utils"

if $PROGRAM_NAME == __FILE__
  options = { seed: 0 }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby encodec/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  config = EncodecExample::EncodecConfig.new(
    audio_channels: 2,
    sampling_rate: 24_000,
    num_filters: 16,
    hidden_size: 32,
    codebook_size: 64,
    codebook_dim: 32,
    upsampling_ratios: [2, 2],
    target_bandwidths: [1.5, 3.0, 6.0],
    num_lstm_layers: 1,
    num_residual_layers: 1,
    chunk_length_s: nil,
    overlap: nil,
    normalize: true
  )

  model = EncodecExample::EncodecModel.new(config)

  audio = MLX::Core.random_uniform([1, 96, 2], -1.0, 1.0, MLX::Core.float32)
  mask = MLX::Core.ones([1, 96], MLX::Core.bool_)

  codes, scales = model.encode(audio, mask, bandwidth: 3.0)
  MLX::Core.eval(codes)
  scales.each { |s| MLX::Core.eval(s) unless s.nil? }

  raise "Expected one encoded frame" unless codes.shape[0] == 1
  raise "Encoded batch size mismatch" unless codes.shape[1] == 1
  raise "Encoded codebook axis must be positive" unless codes.shape[2] > 0
  raise "Encoded frame axis must be positive" unless codes.shape[3] > 0
  raise "Expected scales for each frame" unless scales.length == 1

  decoded = model.decode(codes, scales, mask)
  MLX::Core.eval(decoded)
  raise "Decoded batch mismatch" unless decoded.shape[0] == 1
  raise "Decoded channel mismatch" unless decoded.shape[2] == 2
  raise "Decoded time axis must be positive" unless decoded.shape[1] > 0

  begin
    model.encode(audio, mask, bandwidth: 99.0)
    raise "Expected unsupported bandwidth to raise"
  rescue ArgumentError
    nil
  end

  raw1 = MLX::Core.random_uniform([21, 2], -1.0, 1.0, MLX::Core.float32)
  raw2 = MLX::Core.random_uniform([13, 2], -1.0, 1.0, MLX::Core.float32)
  feats, padding = EncodecExample.preprocess_audio(
    [raw1, raw2],
    sampling_rate: config.sampling_rate,
    chunk_length: 16,
    chunk_stride: 8
  )
  MLX::Core.eval(feats, padding)

  raise "Preprocess batch mismatch" unless feats.shape[0] == 2
  raise "Preprocess channel mismatch" unless feats.shape[2] == 2
  raise "Preprocess padded length mismatch: #{feats.shape[1]}" unless feats.shape[1] == 32

  idx0 = MLX::Core.array([0], MLX::Core.int32)
  idx1 = MLX::Core.array([1], MLX::Core.int32)
  mask0 = MLX::Core.squeeze(MLX::Core.take(padding, idx0, 0), 0)
  mask1 = MLX::Core.squeeze(MLX::Core.take(padding, idx1, 0), 0)
  MLX::Core.eval(mask0, mask1)

  raise "Padding mask length mismatch for sample 0" unless MLX::Core.sum(mask0).item == 21
  raise "Padding mask length mismatch for sample 1" unless MLX::Core.sum(mask1).item == 13

  indices = model.quantizer.layers[0].encode(MLX::Core.random_uniform([1, 8, config.codebook_dim], -1.0, 1.0, MLX::Core.float32))
  restored = model.quantizer.layers[0].decode(indices)
  MLX::Core.eval(indices, restored)
  raise "Codebook index shape mismatch" unless indices.shape == [1, 8]
  raise "Codebook decode shape mismatch" unless restored.shape == [1, 8, config.codebook_dim]

  Dir.mktmpdir("encodec-test-") do |dir|
    wav_path = File.join(dir, "sample.wav")
    clip = MLX::Core.random_uniform([128, 2], -0.5, 0.5, MLX::Core.float32)
    EncodecExample::Utils.save_audio(wav_path, clip, config.sampling_rate)
    raise "Wave file not written" unless File.exist?(wav_path)

    header = File.binread(wav_path, 12)
    raise "Invalid wave RIFF header" unless header.start_with?("RIFF") && header[8, 4] == "WAVE"

    if EncodecExample::Utils.ffmpeg_available?
      loaded = EncodecExample::Utils.load_audio(wav_path, config.sampling_rate, 2)
      MLX::Core.eval(loaded)
      raise "Loaded audio channel mismatch" unless loaded.shape[1] == 2
      raise "Loaded audio empty" unless loaded.shape[0] > 0
    end
  end

  puts "Tests pass :)"
end
