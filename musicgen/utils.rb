# frozen_string_literal: true

require_relative "../encodec/utils"

module MusicGenExample
  module Utils
    module_function

    def save_audio(file, audio, sampling_rate)
      clipped = MLX::Core.clip(audio, -1.0, 1.0)
      EncodecExample::Utils.save_audio(file, clipped, sampling_rate)
    end
  end
end
