# frozen_string_literal: true

require "open3"

require_relative "encodec"

module EncodecExample
  module Utils
    module_function

    def save_audio(file, audio, sampling_rate)
      waveform = audio.respond_to?(:shape) ? audio : MLX::Core.array(audio, MLX::Core.float32)
      data = waveform.to_a

      frames = case waveform.shape.length
               when 1
                 data.map { |sample| [to_pcm_int16(sample)] }
               when 2
                 data.map { |frame| frame.map { |sample| to_pcm_int16(sample) } }
               when 3
                 raise ArgumentError, "Only batch size 1 is supported for save_audio" unless waveform.shape[0] == 1

                 data[0].map { |frame| frame.map { |sample| to_pcm_int16(sample) } }
               else
                 raise ArgumentError, "Unsupported audio shape #{waveform.shape.inspect}"
               end

      channels = frames.empty? ? 1 : frames[0].length
      pcm_bytes = frames.flatten.pack("s<*")
      File.binwrite(file, wav_header(pcm_bytes.bytesize, sampling_rate, channels) + pcm_bytes)
    end

    def load_audio(file, sampling_rate, channels)
      cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file.to_s,
        "-f", "s16le",
        "-ac", channels.to_s,
        "-acodec", "pcm_s16le",
        "-ar", sampling_rate.to_s,
        "-"
      ]

      out, err, status = Open3.capture3(*cmd)
      unless status.success?
        raise "Failed to load audio via ffmpeg: #{err}"
      end

      int16 = out.unpack("s<*")
      samples = int16.map { |v| v.to_f / 32_767.0 }
      frames = samples.each_slice(channels).to_a
      MLX::Core.array(frames, MLX::Core.float32)
    end

    def ffmpeg_available?
      _out, _err, status = Open3.capture3("ffmpeg", "-version")
      status.success?
    rescue Errno::ENOENT
      false
    end

    def to_pcm_int16(sample)
      value = (sample.to_f * 32_767.0).round
      value = -32_768 if value < -32_768
      value = 32_767 if value > 32_767
      value
    end

    def wav_header(data_size, sampling_rate, channels)
      bits_per_sample = 16
      block_align = channels * (bits_per_sample / 8)
      byte_rate = sampling_rate * block_align
      riff_size = 36 + data_size

      header = +""
      header << "RIFF"
      header << [riff_size].pack("V")
      header << "WAVE"
      header << "fmt "
      header << [16].pack("V")
      header << [1].pack("v")
      header << [channels].pack("v")
      header << [sampling_rate].pack("V")
      header << [byte_rate].pack("V")
      header << [block_align].pack("v")
      header << [bits_per_sample].pack("v")
      header << "data"
      header << [data_size].pack("V")
      header
    end
  end
end
