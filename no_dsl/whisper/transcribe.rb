# frozen_string_literal: true

require_relative "audio"
require_relative "decoding"
require_relative "load_models"
require_relative "timing"
require_relative "tokenizer"

module WhisperExample
  module Transcribe
    module_function

    def format_timestamp(seconds)
      ms = (seconds.to_f * 1000.0).round
      hours = ms / 3_600_000
      ms -= hours * 3_600_000
      minutes = ms / 60_000
      ms -= minutes * 60_000
      secs = ms / 1000
      ms -= secs * 1000
      prefix = hours.positive? ? format("%02d:", hours) : ""
      format("%s%02d:%02d.%03d", prefix, minutes, secs, ms)
    end

    def get_end(segments)
      return nil if segments.empty?

      segments[-1]["end"]
    end

    class ModelHolder
      @model = nil
      @model_path = nil

      class << self
        attr_accessor :model, :model_path

        def get_model(path_or_hf_repo, dtype)
          if model.nil? || model_path != [path_or_hf_repo, dtype]
            self.model = LoadModels.load_model(path_or_hf_repo, dtype: dtype)
            self.model_path = [path_or_hf_repo, dtype]
          end
          model
        end
      end
    end

    def transcribe(
      audio,
      path_or_hf_repo: "mlx-community/whisper-tiny",
      verbose: nil,
      temperature: [0.0, 0.2, 0.4],
      compression_ratio_threshold: 2.4,
      logprob_threshold: -1.0,
      no_speech_threshold: 0.6,
      condition_on_previous_text: true,
      initial_prompt: nil,
      word_timestamps: false,
      prepend_punctuations: "\"'“¿([{-",
      append_punctuations: "\"'.。,，!！?？:：”)]}、",
      clip_timestamps: "0",
      hallucination_silence_threshold: nil,
      **decode_options
    )
      _ = compression_ratio_threshold
      _ = logprob_threshold
      _ = no_speech_threshold
      _ = condition_on_previous_text
      _ = initial_prompt
      _ = prepend_punctuations
      _ = append_punctuations
      _ = clip_timestamps
      _ = hallucination_silence_threshold

      dtype = decode_options.fetch(:fp16, true) ? MLX::Core.float16 : MLX::Core.float32
      model = ModelHolder.get_model(path_or_hf_repo, dtype)

      mel = Audio.log_mel_spectrogram(audio, n_mels: model.dims.n_mels, padding: Audio::N_SAMPLES)
      content_frames = [mel.shape[0] - Audio::N_FRAMES, 0].max
      language = decode_options[:language]

      if language.nil?
        if !model.is_multilingual
          language = "en"
        else
          mel_segment = Audio.pad_or_trim(mel, length: model.dims.n_audio_ctx * 2, axis: 0).astype(dtype)
          _token, probs = model.detect_language(mel_segment)
          language = probs.keys.max_by { |k| probs[k] } || "en"
        end
      end

      task = decode_options.fetch(:task, "transcribe")
      tokenizer = TokenizerModule.get_tokenizer(
        model.is_multilingual,
        num_languages: [model.num_languages, 1].max,
        language: language,
        task: task
      )

      temps = temperature.is_a?(Array) ? temperature : [temperature]
      decode_result = nil
      temps.each do |t|
        opts = Decoding::DecodingOptions.new(
          task: task,
          language: language,
          temperature: t,
          fp16: decode_options.fetch(:fp16, true),
          without_timestamps: decode_options.fetch(:without_timestamps, false)
        )
        mel_segment = Audio.pad_or_trim(mel, length: model.dims.n_audio_ctx * 2, axis: 0).astype(dtype)
        decode_result = model.decode(mel_segment, opts)
        break
      end

      end_time = (Audio.pad_or_trim(mel, length: model.dims.n_audio_ctx * 2, axis: 0).shape[0] * Audio::HOP_LENGTH).to_f / Audio::SAMPLE_RATE
      segment = {
        "seek" => 0,
        "start" => 0.0,
        "end" => end_time,
        "text" => decode_result.text,
        "tokens" => decode_result.tokens,
        "temperature" => decode_result.temperature,
        "avg_logprob" => decode_result.avg_logprob,
        "compression_ratio" => decode_result.compression_ratio,
        "no_speech_prob" => decode_result.no_speech_prob
      }

      segments = [segment]
      if word_timestamps
        Timing.add_word_timestamps(
          segments: segments,
          model: model,
          tokenizer: tokenizer,
          mel: mel,
          num_frames: [content_frames, mel.shape[0]].max,
          last_speech_timestamp: 0.0
        )
      end

      puts("[00:00.000 --> #{format_timestamp(segment['end'])}] #{segment['text']}") if verbose

      {
        "text" => " #{decode_result.text}",
        "segments" => segments,
        "language" => language
      }
    end
  end
end
