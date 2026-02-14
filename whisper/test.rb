# frozen_string_literal: true

require "json"
require "optparse"
require "tmpdir"

require_relative "mlx_whisper"
require_relative "convert"
require_relative "cli"

if $PROGRAM_NAME == __FILE__
  options = { seed: 7 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby whisper/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  dims = WhisperExample::ModelDimensions.new(
    n_mels: 32,
    n_audio_ctx: 64,
    n_audio_state: 48,
    n_audio_head: 4,
    n_audio_layer: 2,
    n_vocab: 1024,
    n_text_ctx: 32,
    n_text_state: 48,
    n_text_head: 4,
    n_text_layer: 2
  )

  model = WhisperExample::Whisper.new(dims, dtype: MLX::Core.float32)

  mels = MLX::Core.random_uniform([1, dims.n_audio_ctx * 2, dims.n_mels], -1.0, 1.0, MLX::Core.float32)
  tokens = MLX::Core.array([[10, 11, 12, 13, 14]], MLX::Core.int32)

  logits = model.call(mels, tokens)
  MLX::Core.eval(logits)
  unless logits.shape == [1, tokens.shape[1], dims.n_vocab]
    raise "Forward shape mismatch: expected [1, #{tokens.shape[1]}, #{dims.n_vocab}], got #{logits.shape.inspect}"
  end

  audio_features = model.embed_audio(mels)
  MLX::Core.eval(audio_features)
  unless audio_features.shape == [1, dims.n_audio_ctx, dims.n_audio_state]
    raise "Audio feature shape mismatch: #{audio_features.shape.inspect}"
  end

  lang_token, lang_probs = model.detect_language(MLX::Core.squeeze(mels, 0))
  MLX::Core.eval(lang_token)
  raise "Language probabilities should be a Hash" unless lang_probs.is_a?(Hash)

  decoded = model.decode(
    MLX::Core.squeeze(mels, 0),
    task: "transcribe",
    language: "en",
    temperature: 0.0,
    sample_len: 8,
    fp16: false
  )
  raise "Decoded token sequence is empty" if decoded.tokens.empty?
  raise "Decoded text should be String" unless decoded.text.is_a?(String)

  raw_audio = MLX::Core.random_uniform([12_345], -1.0, 1.0, MLX::Core.float32)
  padded = WhisperExample::Audio.pad_or_trim(raw_audio, length: 16_000)
  raise "pad_or_trim shape mismatch" unless padded.shape == [16_000]

  mel = WhisperExample::Audio.log_mel_spectrogram(raw_audio, n_mels: dims.n_mels, padding: 320)
  MLX::Core.eval(mel)
  raise "Mel feature rank mismatch" unless mel.shape.length == 2
  raise "Mel bins mismatch" unless mel.shape[1] == dims.n_mels

  tokenizer = WhisperExample::TokenizerModule.get_tokenizer(
    true,
    num_languages: 3,
    language: "en",
    task: "transcribe"
  )
  segments = [{ "start" => 0.0, "end" => 1.0, "tokens" => [10, 11, 12], "text" => "abc" }]
  WhisperExample::Timing.add_word_timestamps(
    segments: segments,
    model: model,
    tokenizer: tokenizer,
    mel: mel,
    num_frames: mel.shape[0],
    last_speech_timestamp: 0.0
  )
  raise "Missing words in timestamp output" unless segments[0].key?("words")

  Dir.mktmpdir("whisper-test-") do |dir|
    config = dims.to_h.merge("model_type" => "whisper")
    File.binwrite(File.join(dir, "config.json"), JSON.pretty_generate(config))

    loaded = WhisperExample::LoadModels.load_model(dir, dtype: MLX::Core.float32)
    raise "Loaded model dims mismatch" unless loaded.dims.n_mels == dims.n_mels

    transcribed = WhisperExample.transcribe(
      raw_audio,
      path_or_hf_repo: dir,
      verbose: false,
      language: "en",
      fp16: false,
      word_timestamps: true
    )

    raise "Transcribe result missing text" unless transcribed["text"].is_a?(String)
    raise "Transcribe result missing segments" unless transcribed["segments"].is_a?(Array)
    raise "Transcribe result missing language" unless transcribed["language"].is_a?(String)

    out_dir = File.join(dir, "out")
    Dir.mkdir(out_dir)
    writer = WhisperExample::Writers.get_writer("all", out_dir)
    writer.call(transcribed, "sample")

    %w[txt vtt srt tsv json].each do |ext|
      path = File.join(out_dir, "sample.#{ext}")
      raise "Expected output file #{path}" unless File.exist?(path)
    end

    parsed_json = JSON.parse(File.binread(File.join(out_dir, "sample.json")))
    raise "JSON writer output malformed" unless parsed_json.key?("text")

    mapped = WhisperExample::Convert.hf_to_dims_config(
      "num_mel_bins" => 80,
      "max_source_positions" => 1500,
      "d_model" => 384,
      "encoder_attention_heads" => 6,
      "encoder_layers" => 4,
      "vocab_size" => 51_865,
      "max_target_positions" => 448,
      "decoder_attention_heads" => 6,
      "decoder_layers" => 4
    )
    raise "convert mapping missing n_vocab" unless mapped["n_vocab"] == 51_865
  end

  puts "Tests pass :)"
end
