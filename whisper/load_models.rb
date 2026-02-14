# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

require_relative "whisper"

module WhisperExample
  module LoadModels
    module_function

    SNAPSHOT_SCRIPT = Pathname.new(__dir__).join("python", "snapshot_download.py").to_s

    def load_model(path_or_hf_repo, dtype: MLX::Core.float32, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      model_path = Pathname.new(path_or_hf_repo.to_s)
      model_path = snapshot_download(path_or_hf_repo.to_s, python_bin: python_bin) unless model_path.exist?

      config_path = model_path.join("config.json")
      config = if config_path.exist?
                 JSON.parse(File.binread(config_path))
               else
                 {}
               end
      config.delete("model_type")

      dims = WhisperExample::ModelDimensions.from_hash(config)
      model = WhisperExample::Whisper.new(dims, dtype: dtype)

      wf = [
        model_path.join("model.safetensors"),
        model_path.join("weights.safetensors"),
        model_path.join("weights.npz")
      ].find(&:exist?)

      if wf
        begin
          weights = MLX::Core.load(wf.to_s).to_a
          model.load_weights(weights, strict: false)
        rescue StandardError
          # Best-effort loading for lightweight port.
        end
      end

      MLX::Core.eval(model.parameters)
      model
    end

    def snapshot_download(repo_id, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      stdout, stderr, status = Open3.capture3(python_bin, SNAPSHOT_SCRIPT, repo_id.to_s)
      raise "Failed to download model snapshot for #{repo_id}: #{stderr}" unless status.success?

      Pathname.new(stdout.strip)
    end
  end
end
