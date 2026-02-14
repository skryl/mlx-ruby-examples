# frozen_string_literal: true

require "optparse"
require "tmpdir"

require_relative "musicgen"
require_relative "utils"

if $PROGRAM_NAME == __FILE__
  options = { seed: 123 }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby musicgen/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  config = MusicGenExample::MusicGenConfig.new(
    text_encoder: MusicGenExample::TextEncoderConfig.new(
      _name_or_path: "toy-text-encoder",
      d_model: 32,
      vocab_size: 256,
      max_length: 12
    ),
    audio_encoder: MusicGenExample::AudioEncoderConfig.new(
      _name_or_path: "toy-audio-encoder",
      codebook_size: 32,
      sampling_rate: 24_000
    ),
    decoder: MusicGenExample::DecoderConfig.new(
      num_codebooks: 4,
      bos_token_id: 32,
      hidden_size: 64,
      num_attention_heads: 4,
      ffn_dim: 128,
      num_hidden_layers: 2
    )
  )

  model = MusicGenExample::MusicGen.new(config)

  conditioning = model.text_conditioner.call("folk ballad")
  MLX::Core.eval(conditioning)
  raise "Conditioning batch mismatch" unless conditioning.shape[0] == 1
  raise "Conditioning hidden mismatch" unless conditioning.shape[2] == config.decoder.hidden_size

  audio_tokens = MLX::Core.full([1, 3, config.decoder.num_codebooks], config.decoder.bos_token_id, MLX::Core.int32)
  logits = model.call(audio_tokens, conditioning)
  MLX::Core.eval(logits)
  expected_shape = [1, 3, config.audio_encoder.codebook_size, config.decoder.num_codebooks]
  unless logits.shape == expected_shape
    raise "Forward shape mismatch: expected #{expected_shape.inspect}, got #{logits.shape.inspect}"
  end

  head_dim = config.decoder.hidden_size / config.decoder.num_attention_heads
  cache = Array.new(config.decoder.num_hidden_layers) { MusicGenExample::KVCache.new(head_dim, config.decoder.num_attention_heads) }
  step0 = MLX::Core.slice(audio_tokens, [0, 0, 0], [1, 1, config.decoder.num_codebooks])
  step1 = MLX::Core.slice(audio_tokens, [0, 1, 0], [1, 2, config.decoder.num_codebooks])
  _logits0 = model.call(step0, conditioning, cache: cache)
  _logits1 = model.call(step1, conditioning, cache: cache)
  raise "Cache offset mismatch" unless cache[0].offset == 2

  sampled = MusicGenExample.top_k_sampling(logits, 8, 1.0, axis: -2)
  MLX::Core.eval(sampled)
  raise "Sampled token shape mismatch" unless sampled.shape == [1, 3, 1, config.decoder.num_codebooks]

  pos_emb = MusicGenExample.create_sin_embedding(5, config.decoder.hidden_size)
  MLX::Core.eval(pos_emb)
  raise "Sinusoidal embedding shape mismatch" unless pos_emb.shape == [1, 1, config.decoder.hidden_size]

  generated = model.generate("tiny synth melody", max_steps: 4, top_k: 8, temp: 1.0, guidance_coef: 2.0)
  MLX::Core.eval(generated)
  raise "Generated audio rank mismatch" unless generated.shape.length == 2
  raise "Generated audio channel mismatch" unless generated.shape[1] == 1
  raise "Generated audio should be non-empty" unless generated.shape[0] > 0

  Dir.mktmpdir("musicgen-test-") do |dir|
    path = File.join(dir, "out.wav")
    MusicGenExample::Utils.save_audio(path, generated, config.audio_encoder.sampling_rate)
    raise "Expected saved wave file" unless File.exist?(path)

    header = File.binread(path, 12)
    raise "Invalid wave header" unless header.start_with?("RIFF") && header[8, 4] == "WAVE"
  end

  puts "Tests pass :)"
end
