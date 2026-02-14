# frozen_string_literal: true

require "json"
require "pathname"

module WhisperExample
  module Writers
    module_function

    def format_timestamp(seconds, always_include_hours: false, decimal_marker: ".")
      raise ArgumentError, "non-negative timestamp expected" if seconds.to_f.negative?

      ms = (seconds.to_f * 1000.0).round
      hours = ms / 3_600_000
      ms -= hours * 3_600_000
      minutes = ms / 60_000
      ms -= minutes * 60_000
      secs = ms / 1000
      ms -= secs * 1000

      hours_marker = always_include_hours || hours.positive? ? format("%02d:", hours) : ""
      format("%s%02d:%02d%s%03d", hours_marker, minutes, secs, decimal_marker, ms)
    end

    def get_start(segments)
      return nil if segments.nil? || segments.empty?

      segments[0]["start"]
    end

    class ResultWriter
      attr_reader :output_dir

      def initialize(output_dir)
        @output_dir = output_dir
      end

      def extension
        raise NotImplementedError
      end

      def call(result, output_name, options: nil, **kwargs)
        path = Pathname.new(output_dir).join("#{output_name}.#{extension}")
        File.open(path, "w:utf-8") do |f|
          write_result(result, file: f, options: options, **kwargs)
        end
      end

      def write_result(_result, file:, options: nil, **kwargs)
        _ = file
        _ = options
        _ = kwargs
        raise NotImplementedError
      end
    end

    class WriteTXT < ResultWriter
      def extension
        "txt"
      end

      def write_result(result, file:, options: nil, **kwargs)
        _ = options
        _ = kwargs
        result.fetch("segments", []).each do |segment|
          file.puts(segment.fetch("text", "").strip)
        end
      end
    end

    class SubtitlesWriter < ResultWriter
      def always_include_hours
        false
      end

      def decimal_marker
        "."
      end

      def iterate_result(result)
        result.fetch("segments", []).map do |segment|
          [
            Writers.format_timestamp(segment.fetch("start", 0.0), always_include_hours: always_include_hours, decimal_marker: decimal_marker),
            Writers.format_timestamp(segment.fetch("end", 0.0), always_include_hours: always_include_hours, decimal_marker: decimal_marker),
            segment.fetch("text", "").strip.gsub("-->", "->")
          ]
        end
      end
    end

    class WriteVTT < SubtitlesWriter
      def extension
        "vtt"
      end

      def write_result(result, file:, options: nil, **kwargs)
        _ = options
        _ = kwargs
        file.puts("WEBVTT\n\n")
        iterate_result(result).each do |start_t, end_t, text|
          file.puts("#{start_t} --> #{end_t}")
          file.puts(text)
          file.puts
        end
      end
    end

    class WriteSRT < SubtitlesWriter
      def extension
        "srt"
      end

      def always_include_hours
        true
      end

      def decimal_marker
        ","
      end

      def write_result(result, file:, options: nil, **kwargs)
        _ = options
        _ = kwargs
        iterate_result(result).each_with_index do |(start_t, end_t, text), idx|
          file.puts(idx + 1)
          file.puts("#{start_t} --> #{end_t}")
          file.puts(text)
          file.puts
        end
      end
    end

    class WriteTSV < ResultWriter
      def extension
        "tsv"
      end

      def write_result(result, file:, options: nil, **kwargs)
        _ = options
        _ = kwargs
        file.puts("start\tend\ttext")
        result.fetch("segments", []).each do |segment|
          start_ms = (segment.fetch("start", 0.0).to_f * 1000).round
          end_ms = (segment.fetch("end", 0.0).to_f * 1000).round
          text = segment.fetch("text", "").gsub("\t", " ").strip
          file.puts("#{start_ms}\t#{end_ms}\t#{text}")
        end
      end
    end

    class WriteJSON < ResultWriter
      def extension
        "json"
      end

      def write_result(result, file:, options: nil, **kwargs)
        _ = options
        _ = kwargs
        file.write(JSON.pretty_generate(result))
      end
    end

    WRITERS = {
      "txt" => WriteTXT,
      "vtt" => WriteVTT,
      "srt" => WriteSRT,
      "tsv" => WriteTSV,
      "json" => WriteJSON
    }.freeze

    def get_writer(output_format, output_dir)
      if output_format == "all"
        writer_instances = WRITERS.values.map { |klass| klass.new(output_dir) }
        lambda do |result, output_name, options: nil, **kwargs|
          writer_instances.each do |writer|
            writer.call(result, output_name, options: options, **kwargs)
          end
        end
      else
        klass = WRITERS.fetch(output_format) do
          raise ArgumentError, "Unsupported output format '#{output_format}'"
        end
        klass.new(output_dir)
      end
    end
  end
end
