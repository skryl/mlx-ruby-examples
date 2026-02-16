# frozen_string_literal: true

require_relative "_version"
require_relative "audio"
require_relative "decoding"
require_relative "load_models"
require_relative "timing"
require_relative "tokenizer"
require_relative "transcribe"
require_relative "whisper"
require_relative "writers"

module WhisperExample
  module_function

  def transcribe(*args, **kwargs)
    Transcribe.transcribe(*args, **kwargs)
  end
end
