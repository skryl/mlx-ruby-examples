# frozen_string_literal: true

require "optparse"

require_relative "encodec"
require_relative "utils"

if $PROGRAM_NAME == __FILE__
  options = {
    model: "mlx-community/encodec-48khz-float32",
    bandwidth: 3.0,
    input: nil,
    output: "reconstructed.wav",
    seconds: 1.0,
    seed: 0
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby encodec/example.rb [options]"
    opts.on("--model REPO", String, "Local model path or Hugging Face repo") { |v| options[:model] = v }
    opts.on("--bandwidth N", Float, "Target bandwidth in kbps") { |v| options[:bandwidth] = v }
    opts.on("--input FILE", String, "Input audio file (optional, uses synthetic audio if omitted)") { |v| options[:input] = v }
    opts.on("--output FILE", String, "Output wave file path") { |v| options[:output] = v }
    opts.on("--seconds N", Float, "Synthetic audio length in seconds") { |v| options[:seconds] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  model, processor = EncodecExample::EncodecModel.from_pretrained(options[:model])

  audio = if options[:input]
            EncodecExample::Utils.load_audio(options[:input], model.sampling_rate, model.channels)
          else
            sample_count = (model.sampling_rate * options[:seconds]).to_i
            MLX::Core.random_uniform([sample_count, model.channels], -1.0, 1.0, MLX::Core.float32)
          end

  feats, mask = processor.call(audio)
  codes, scales = model.encode(feats, mask, bandwidth: options[:bandwidth])
  reconstructed = model.decode(codes, scales, mask)

  target_length = audio.shape[0]
  reconstructed = MLX::Core.slice(
    reconstructed,
    [0, 0, 0],
    [reconstructed.shape[0], [target_length, reconstructed.shape[1]].min, reconstructed.shape[2]]
  )
  reconstructed = MLX::Core.squeeze(reconstructed, 0)

  EncodecExample::Utils.save_audio(options[:output], reconstructed, model.sampling_rate)
  puts "Wrote #{options[:output]}"
end
