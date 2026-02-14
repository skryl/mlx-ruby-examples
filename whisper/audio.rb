# frozen_string_literal: true

require "open3"

module WhisperExample
  module Audio
    module_function

    SAMPLE_RATE = 16_000
    N_FFT = 400
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
    N_FRAMES = N_SAMPLES / HOP_LENGTH

    N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
    FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH
    TOKENS_PER_SECOND = SAMPLE_RATE / N_SAMPLES_PER_TOKEN

    def load_audio(file = nil, sr: SAMPLE_RATE, from_stdin: false)
      if from_stdin
        stdin_data = $stdin.read
        cmd = ["ffmpeg", "-i", "pipe:0"]
        cmd.concat(common_ffmpeg_tail(sr))
        out, err, status = Open3.capture3(*cmd, stdin_data: stdin_data)
      else
        raise ArgumentError, "audio file path is required" if file.nil?

        cmd = ["ffmpeg", "-nostdin", "-i", file.to_s]
        cmd.concat(common_ffmpeg_tail(sr))
        out, err, status = Open3.capture3(*cmd)
      end

      raise "Failed to load audio: #{err}" unless status.success?

      samples = out.unpack("s<*").map { |v| v.to_f / 32_768.0 }
      MLX::Core.array(samples, MLX::Core.float32)
    rescue Errno::ENOENT
      raise "ffmpeg is required but not found in PATH"
    end

    def pad_or_trim(array, length: N_SAMPLES, axis: -1)
      x = array.respond_to?(:shape) ? array : MLX::Core.array(array, MLX::Core.float32)
      ndim = x.shape.length
      axis = ndim + axis if axis.negative?

      current = x.shape[axis]
      if current > length
        x = slice_axis(x, axis, 0, length)
      elsif current < length
        pad_widths = Array.new(ndim) { [0, 0] }
        pad_widths[axis] = [0, length - current]
        x = MLX::Core.pad(x, pad_widths)
      end
      x
    end

    def log_mel_spectrogram(audio, n_mels: 80, padding: 0)
      waveform = if audio.is_a?(String)
                   load_audio(audio)
                 elsif audio.respond_to?(:shape)
                   audio
                 else
                   MLX::Core.array(audio, MLX::Core.float32)
                 end

      waveform = waveform.astype(MLX::Core.float32)
      waveform = MLX::Core.pad(waveform, [[0, padding]]) if padding.positive?

      # Lightweight mel-like feature extractor: frame energy expanded into n_mels bins.
      total = waveform.shape[0]
      frame_count = [total / HOP_LENGTH, 1].max
      trimmed = slice_axis(waveform, 0, 0, frame_count * HOP_LENGTH)
      frames = MLX::Core.reshape(trimmed, [frame_count, HOP_LENGTH])
      energy = MLX::Core.mean(MLX::Core.abs(frames), 1)
      mel_axis = MLX::Core.reshape(MLX::Core.arange(0, n_mels, 1, MLX::Core.float32), [1, n_mels])
      mel = MLX::Core.multiply(MLX::Core.expand_dims(energy, 1), MLX::Core.add(1.0, MLX::Core.divide(mel_axis, n_mels.to_f)))
      mel = MLX::Core.log10(MLX::Core.add(mel, 1e-5))
      mel
    end

    def common_ffmpeg_tail(sr)
      [
        "-threads", "0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", sr.to_s,
        "-"
      ]
    end

    def slice_axis(x, axis, start_idx, end_idx)
      case x.shape.length
      when 1
        MLX::Core.slice(x, [start_idx], [end_idx])
      when 2
        if axis == 0
          MLX::Core.slice(x, [start_idx, 0], [end_idx, x.shape[1]])
        else
          MLX::Core.slice(x, [0, start_idx], [x.shape[0], end_idx])
        end
      when 3
        if axis == 0
          MLX::Core.slice(x, [start_idx, 0, 0], [end_idx, x.shape[1], x.shape[2]])
        elsif axis == 1
          MLX::Core.slice(x, [0, start_idx, 0], [x.shape[0], end_idx, x.shape[2]])
        else
          MLX::Core.slice(x, [0, 0, start_idx], [x.shape[0], x.shape[1], end_idx])
        end
      else
        raise ArgumentError, "slice_axis supports up to rank-3 tensors"
      end
    end
  end
end
