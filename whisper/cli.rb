# frozen_string_literal: true

require "optparse"
require "pathname"

require_relative "audio"
require_relative "tokenizer"
require_relative "transcribe"
require_relative "writers"

module WhisperExample
  module CLI
    module_function

    def build_parser(options)
      OptionParser.new do |opts|
        opts.banner = "Usage: ruby whisper/cli.rb [options] audio1 [audio2 ...]"

        opts.on("--model MODEL", String, "Model path or HF repo") { |v| options[:model] = v }
        opts.on("--output-name NAME", String, "Output base file name") { |v| options[:output_name] = v }
        opts.on("--output-dir DIR", String, "Output directory") { |v| options[:output_dir] = v }
        opts.on("--output-format FORMAT", String, "txt|vtt|srt|tsv|json|all") { |v| options[:output_format] = v }
        opts.on("--verbose BOOL", String, "Verbose output (true/false)") { |v| options[:verbose] = v.downcase == "true" }
        opts.on("--task TASK", String, "transcribe|translate") { |v| options[:task] = v }
        opts.on("--language LANG", String, "Language code") { |v| options[:language] = v }
        opts.on("--temperature N", Float, "Decoding temperature") { |v| options[:temperature] = v }
        opts.on("--word-timestamps BOOL", String, "Enable word timestamps") { |v| options[:word_timestamps] = v.downcase == "true" }
        opts.on("--fp16 BOOL", String, "Use float16") { |v| options[:fp16] = v.downcase == "true" }
      end
    end

    def main(argv = ARGV)
      options = {
        model: "mlx-community/whisper-tiny",
        output_name: nil,
        output_dir: ".",
        output_format: "txt",
        verbose: true,
        task: "transcribe",
        language: nil,
        temperature: 0.0,
        word_timestamps: false,
        fp16: true
      }

      parser = build_parser(options)
      parser.parse!(argv)
      audio_inputs = argv
      if audio_inputs.empty?
        warn parser.to_s
        return 2
      end

      Dir.mkdir(options[:output_dir]) unless Dir.exist?(options[:output_dir])
      writer = Writers.get_writer(options[:output_format], options[:output_dir])

      audio_inputs.each do |audio_obj|
        output_name = options[:output_name] || (audio_obj == "-" ? "content" : Pathname.new(audio_obj).basename.sub_ext("").to_s)
        input_audio = audio_obj == "-" ? Audio.load_audio(from_stdin: true) : audio_obj
        result = Transcribe.transcribe(
          input_audio,
          path_or_hf_repo: options[:model],
          verbose: options[:verbose],
          task: options[:task],
          language: options[:language],
          temperature: options[:temperature],
          word_timestamps: options[:word_timestamps],
          fp16: options[:fp16]
        )
        writer.call(result, output_name)
      end

      0
    end
  end
end

if $PROGRAM_NAME == __FILE__
  exit(WhisperExample::CLI.main(ARGV))
end
