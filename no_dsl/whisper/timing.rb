# frozen_string_literal: true

require_relative "audio"

module WhisperExample
  module Timing
    module_function

    WordTiming = Struct.new(:word, :tokens, :start, :end, :probability, keyword_init: true)

    def median_filter(x, _filter_width)
      x
    end

    def dtw(x)
      n = [x.shape[0], x.shape[1]].min
      idx = (0...n).to_a
      [idx, idx]
    end

    def find_alignment(model, tokenizer, text_tokens, mel, num_frames, medfilt_width: 7, qk_scale: 1.0)
      _ = model
      _ = tokenizer
      _ = mel
      _ = num_frames
      _ = medfilt_width
      _ = qk_scale

      words = tokenizer.decode(text_tokens).split(/(\s+)/).reject(&:empty?)
      return [] if words.empty?

      duration = [text_tokens.length / Audio::TOKENS_PER_SECOND.to_f, 0.1].max
      step = duration / words.length
      current = 0.0
      words.map do |w|
        wt = WordTiming.new(
          word: w,
          tokens: tokenizer.encode(w),
          start: current,
          end: current + step,
          probability: 0.5
        )
        current += step
        wt
      end
    end

    def merge_punctuations(alignment, prepended, appended)
      _ = prepended
      _ = appended
      alignment
    end

    def add_word_timestamps(
      segments:,
      model:,
      tokenizer:,
      mel:,
      num_frames:,
      prepend_punctuations: "\"'“¿([{-",
      append_punctuations: "\"'.。,，!！?？:：”)]}、",
      last_speech_timestamp: 0.0,
      **kwargs
    )
      _ = model
      _ = mel
      _ = num_frames
      _ = last_speech_timestamp
      _ = kwargs

      segments.each do |segment|
        text_tokens = segment.fetch("tokens", []).select { |t| t.is_a?(Integer) && t < tokenizer.eot }
        alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames)
        merge_punctuations(alignment, prepend_punctuations, append_punctuations)

        duration = [segment["end"].to_f - segment["start"].to_f, 0.1].max
        if alignment.empty?
          segment["words"] = []
          next
        end

        step = duration / alignment.length
        cursor = segment["start"].to_f
        segment["words"] = alignment.map do |w|
          start_t = cursor
          end_t = [segment["end"].to_f, cursor + step].min
          cursor = end_t
          {
            "word" => w.word,
            "start" => start_t,
            "end" => end_t,
            "probability" => w.probability
          }
        end
      end
    end
  end
end
