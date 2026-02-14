# frozen_string_literal: true

require "optparse"

require_relative "musicgen"
require_relative "utils"

if $PROGRAM_NAME == __FILE__
  options = {
    model: "facebook/musicgen-medium",
    text: "happy rock",
    output_path: "0.wav",
    max_steps: 500,
    top_k: 250,
    temp: 1.0,
    guidance: 3.0
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby musicgen/generate.rb [options]"
    opts.on("--model NAME", String, "Local model path or HF repo") { |v| options[:model] = v }
    opts.on("--text TEXT", String, "Conditioning prompt") { |v| options[:text] = v }
    opts.on("--output-path FILE", String, "Output wav path") { |v| options[:output_path] = v }
    opts.on("--max-steps N", Integer, "Number of autoregressive decoding steps") { |v| options[:max_steps] = v }
    opts.on("--top-k N", Integer, "Top-k decoding") { |v| options[:top_k] = v }
    opts.on("--temp N", Float, "Sampling temperature") { |v| options[:temp] = v }
    opts.on("--guidance N", Float, "Classifier-free guidance coefficient") { |v| options[:guidance] = v }
  end
  parser.parse!

  model = MusicGenExample::MusicGen.from_pretrained(options[:model])
  audio = model.generate(
    options[:text],
    max_steps: options[:max_steps],
    top_k: options[:top_k],
    temp: options[:temp],
    guidance_coef: options[:guidance]
  )

  MusicGenExample::Utils.save_audio(options[:output_path], audio, model.sampling_rate)
  puts "Wrote #{options[:output_path]}"
end
