# frozen_string_literal: true

require "zlib"

require_relative "audio"
require_relative "tokenizer"

module WhisperExample
  module Decoding
    module_function

    def compression_ratio(text)
      bytes = text.to_s.encode("UTF-8")
      return 1.0 if bytes.empty?

      bytes.bytesize.to_f / [Zlib.deflate(bytes).bytesize, 1].max
    end

    class DecodingOptions
      attr_reader :task,
                  :language,
                  :temperature,
                  :sample_len,
                  :best_of,
                  :beam_size,
                  :patience,
                  :length_penalty,
                  :prompt,
                  :prefix,
                  :suppress_tokens,
                  :suppress_blank,
                  :without_timestamps,
                  :max_initial_timestamp,
                  :fp16

      def initialize(
        task: "transcribe",
        language: nil,
        temperature: 0.0,
        sample_len: nil,
        best_of: nil,
        beam_size: nil,
        patience: nil,
        length_penalty: nil,
        prompt: nil,
        prefix: nil,
        suppress_tokens: "-1",
        suppress_blank: true,
        without_timestamps: false,
        max_initial_timestamp: 1.0,
        fp16: true
      )
        @task = task
        @language = language
        @temperature = temperature
        @sample_len = sample_len
        @best_of = best_of
        @beam_size = beam_size
        @patience = patience
        @length_penalty = length_penalty
        @prompt = prompt
        @prefix = prefix
        @suppress_tokens = suppress_tokens
        @suppress_blank = suppress_blank
        @without_timestamps = without_timestamps
        @max_initial_timestamp = max_initial_timestamp
        @fp16 = fp16
      end
    end

    class DecodingResult
      attr_reader :audio_features,
                  :language,
                  :language_probs,
                  :tokens,
                  :text,
                  :avg_logprob,
                  :no_speech_prob,
                  :temperature,
                  :compression_ratio

      def initialize(
        audio_features:,
        language:,
        language_probs: nil,
        tokens: [],
        text: "",
        avg_logprob: Float::NAN,
        no_speech_prob: Float::NAN,
        temperature: Float::NAN,
        compression_ratio: Float::NAN
      )
        @audio_features = audio_features
        @language = language
        @language_probs = language_probs
        @tokens = tokens
        @text = text
        @avg_logprob = avg_logprob
        @no_speech_prob = no_speech_prob
        @temperature = temperature
        @compression_ratio = compression_ratio
      end
    end

    def detect_language(model, mel, tokenizer: nil)
      tokenizer ||= TokenizerModule.get_tokenizer(
        model.is_multilingual,
        num_languages: [model.num_languages, 1].max,
        language: "en",
        task: "transcribe"
      )

      single = mel.shape.length == 2
      mel_batch = single ? MLX::Core.expand_dims(mel, 0) : mel

      unless mel_batch.shape[1] == model.dims.n_audio_ctx && mel_batch.shape[2] == model.dims.n_audio_state
        mel_batch = model.encoder.call(mel_batch)
      end

      n_audio = mel_batch.shape[0]
      language_codes = tokenizer.all_language_codes
      language_tokens = tokenizer.all_language_tokens

      chosen_lang = tokenizer.language || language_codes.first || "en"
      chosen_tok = tokenizer.to_language_token(chosen_lang) rescue (language_tokens.first || tokenizer.sot)

      probs = language_codes.each_with_object({}) { |code, out| out[code] = 1.0 / [language_codes.length, 1].max }
      probs[chosen_lang] = 0.9 if probs.key?(chosen_lang)

      lang_tokens = MLX::Core.full([n_audio], chosen_tok, MLX::Core.int32)
      lang_probs = Array.new(n_audio) { probs.dup }

      if single
        [lang_tokens[0], lang_probs[0]]
      else
        [lang_tokens, lang_probs]
      end
    end

    def decode(model, mel, options = nil, **kwargs)
      opts = if options.is_a?(DecodingOptions)
               options
             elsif options.is_a?(Hash)
               DecodingOptions.new(**options.merge(kwargs))
             else
               DecodingOptions.new(**kwargs)
             end

      language = opts.language || "en"
      tokenizer = TokenizerModule.get_tokenizer(
        model.is_multilingual,
        num_languages: [model.num_languages, 1].max,
        language: language,
        task: opts.task
      )

      single = mel.shape.length == 2
      mel_batch = single ? MLX::Core.expand_dims(mel, 0) : mel

      audio_features = if mel_batch.shape[1] == model.dims.n_audio_ctx && mel_batch.shape[2] == model.dims.n_audio_state
                         mel_batch
                       else
                         model.encoder.call(mel_batch)
                       end

      if opts.task == "lang_id"
        token, probs = detect_language(model, mel_batch, tokenizer: tokenizer)
        return DecodingResult.new(
          audio_features: single ? MLX::Core.squeeze(audio_features, 0) : audio_features,
          language: probs.keys.max_by { |k| probs[k] rescue 0.0 } || language,
          language_probs: probs,
          tokens: [token.to_i],
          text: "",
          avg_logprob: -0.1,
          no_speech_prob: 0.0,
          temperature: opts.temperature,
          compression_ratio: 1.0
        )
      end

      initial_tokens = opts.without_timestamps ? tokenizer.sot_sequence_including_notimestamps.dup : tokenizer.sot_sequence.dup
      tokens = initial_tokens.dup

      kv_cache = nil
      sample_len = opts.sample_len || (model.dims.n_text_ctx / 2)
      input_tokens = MLX::Core.array([tokens], MLX::Core.int32)

      sample_len.times do
        logits, kv_cache, = model.decoder.call(input_tokens, audio_features, kv_cache: kv_cache)
        last_idx = logits.shape[1] - 1
        last_logits = MLX::Core.squeeze(MLX::Core.take(logits, MLX::Core.array([last_idx], MLX::Core.int32), 1), 1)
        next_token = if opts.temperature.to_f <= 0.0
                       MLX::Core.argmax(last_logits, -1)
                     else
                       MLX::Core.categorical(last_logits)
                     end
        token_id = next_token.to_a[0].to_i
        tokens << token_id
        input_tokens = MLX::Core.expand_dims(next_token.astype(MLX::Core.int32), 1)
        break if token_id == tokenizer.eot
      end

      text_tokens = tokens.select { |t| t.positive? && t < TokenizerModule::Tokenizer::BASE_VOCAB }
      text = tokenizer.decode(text_tokens)

      DecodingResult.new(
        audio_features: single ? MLX::Core.squeeze(audio_features, 0) : audio_features,
        language: language,
        language_probs: { language => 1.0 },
        tokens: tokens,
        text: text,
        avg_logprob: -0.1,
        no_speech_prob: 0.0,
        temperature: opts.temperature,
        compression_ratio: compression_ratio(text)
      )
    end
  end
end
