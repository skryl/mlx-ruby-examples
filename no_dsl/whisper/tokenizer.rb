# frozen_string_literal: true

module WhisperExample
  module TokenizerModule
    module_function

    LANGUAGES = {
      "en" => "english",
      "de" => "german",
      "es" => "spanish",
      "fr" => "french",
      "it" => "italian",
      "pt" => "portuguese",
      "nl" => "dutch",
      "ja" => "japanese",
      "zh" => "chinese",
      "ko" => "korean"
    }.freeze

    TO_LANGUAGE_CODE = LANGUAGES.each_with_object({}) { |(k, v), out| out[v] = k }.merge(
      "castilian" => "es",
      "mandarin" => "zh"
    ).freeze

    class Tokenizer
      attr_reader :num_languages,
                  :language,
                  :task,
                  :sot_sequence,
                  :special_tokens

      BASE_VOCAB = 50_000
      EOT_TOKEN = 50_257

      def initialize(num_languages:, language: nil, task: nil)
        @num_languages = num_languages
        @language = language
        @task = task

        @special_tokens = {}
        @special_tokens["<|endoftext|>"] = EOT_TOKEN
        @special_tokens["<|startoftranscript|>"] = 50_300

        LANGUAGES.keys.first(num_languages).each_with_index do |lang, idx|
          @special_tokens["<|#{lang}|>"] = 50_301 + idx
        end

        @special_tokens["<|translate|>"] = 50_401
        @special_tokens["<|transcribe|>"] = 50_402
        @special_tokens["<|startoflm|>"] = 50_403
        @special_tokens["<|startofprev|>"] = 50_404
        @special_tokens["<|nospeech|>"] = 50_405
        @special_tokens["<|notimestamps|>"] = 50_406
        @special_tokens["<|0.00|>"] = 52_000

        sot = @special_tokens["<|startoftranscript|>"]
        seq = [sot]
        seq << language_token if !language.nil? && LANGUAGES.key?(language)
        seq << (task == "translate" ? translate : transcribe) unless task.nil?
        @sot_sequence = seq.freeze
      end

      def encode(text)
        text.to_s.bytes.map { |b| (b % 255) + 1 }
      end

      def decode(token_ids)
        ids = token_ids.map(&:to_i).select { |t| t.positive? && t < BASE_VOCAB }
        ids.map(&:chr).join
      rescue StandardError
        ""
      end

      def decode_with_timestamps(token_ids)
        decode(token_ids)
      end

      def eot
        @special_tokens["<|endoftext|>"]
      end

      def transcribe
        @special_tokens["<|transcribe|>"]
      end

      def translate
        @special_tokens["<|translate|>"]
      end

      def sot
        @special_tokens["<|startoftranscript|>"]
      end

      def sot_lm
        @special_tokens["<|startoflm|>"]
      end

      def sot_prev
        @special_tokens["<|startofprev|>"]
      end

      def no_speech
        @special_tokens["<|nospeech|>"]
      end

      def no_timestamps
        @special_tokens["<|notimestamps|>"]
      end

      def timestamp_begin
        @special_tokens["<|0.00|>"]
      end

      def language_token
        raise "language is not configured" if language.nil?

        to_language_token(language)
      end

      def to_language_token(lang)
        tok = @special_tokens["<|#{lang}|>"]
        raise KeyError, "Language #{lang} not found" if tok.nil?

        tok
      end

      def all_language_tokens
        LANGUAGES.keys.first(num_languages).map { |lang| @special_tokens["<|#{lang}|>"] }
      end

      def all_language_codes
        LANGUAGES.keys.first(num_languages)
      end

      def sot_sequence_including_notimestamps
        (sot_sequence + [no_timestamps]).freeze
      end

      def non_speech_tokens
        [encode("[").first, encode("]").first, encode("(").first, encode(")").first]
      end

      def split_to_word_tokens(tokens)
        text = decode(tokens)
        words = text.split(/(\s+)/).reject(&:empty?)
        word_tokens = words.map { |w| encode(w) }
        [words, word_tokens]
      end
    end

    def get_tokenizer(is_multilingual, num_languages: 1, language: nil, task: nil)
      lang = is_multilingual ? language : "en"
      Tokenizer.new(num_languages: num_languages, language: lang, task: task)
    end
  end
end
