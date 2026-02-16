# frozen_string_literal: true

require "json"
require "optparse"
require "open3"
require "pathname"

require_relative "load_models"

module WhisperExample
  module Convert
    module_function

    MODELS = {
      "tiny" => "openai/whisper-tiny",
      "tiny.en" => "openai/whisper-tiny.en",
      "base" => "openai/whisper-base",
      "base.en" => "openai/whisper-base.en",
      "small" => "openai/whisper-small",
      "small.en" => "openai/whisper-small.en",
      "medium" => "openai/whisper-medium",
      "medium.en" => "openai/whisper-medium.en",
      "large-v3" => "openai/whisper-large-v3",
      "turbo" => "openai/whisper-large-v3-turbo"
    }.freeze

    EXTRACT_SCRIPT = Pathname.new(__dir__).join("python", "extract_torch_weights.py").to_s

    def available_models
      MODELS.keys
    end

    def normalize_repo(name_or_path)
      MODELS.fetch(name_or_path.to_s, name_or_path.to_s)
    end

    def hf_to_dims_config(config)
      data = config.transform_keys(&:to_s)
      if data.key?("n_mels")
        {
          "n_mels" => data.fetch("n_mels"),
          "n_audio_ctx" => data.fetch("n_audio_ctx"),
          "n_audio_state" => data.fetch("n_audio_state"),
          "n_audio_head" => data.fetch("n_audio_head"),
          "n_audio_layer" => data.fetch("n_audio_layer"),
          "n_vocab" => data.fetch("n_vocab"),
          "n_text_ctx" => data.fetch("n_text_ctx"),
          "n_text_state" => data.fetch("n_text_state"),
          "n_text_head" => data.fetch("n_text_head"),
          "n_text_layer" => data.fetch("n_text_layer")
        }
      else
        {
          "n_mels" => data.fetch("num_mel_bins", 80),
          "n_audio_ctx" => data.fetch("max_source_positions", 1500),
          "n_audio_state" => data.fetch("d_model", 384),
          "n_audio_head" => data.fetch("encoder_attention_heads", 6),
          "n_audio_layer" => data.fetch("encoder_layers", 4),
          "n_vocab" => data.fetch("vocab_size", 51_865),
          "n_text_ctx" => data.fetch("max_target_positions", 448),
          "n_text_state" => data.fetch("d_model", 384),
          "n_text_head" => data.fetch("decoder_attention_heads", 6),
          "n_text_layer" => data.fetch("decoder_layers", 4)
        }
      end
    end

    def cast_dtype(weights, dtype)
      type = case dtype
             when "float16" then MLX::Core.float16
             when "float32" then MLX::Core.float32
             else
               raise ArgumentError, "dtype must be float16 or float32"
             end
      weights.transform_values { |v| v.astype(type) }
    end

    def extract_torch_bin(bin_path, out_npz, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      stdout, stderr, status = Open3.capture3(python_bin, EXTRACT_SCRIPT, bin_path.to_s, out_npz.to_s)
      raise "Failed to extract torch weights: #{stderr}" unless status.success?

      Pathname.new(stdout.strip)
    end

    def convert(name_or_path, mlx_path:, dtype: "float16", python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      source = normalize_repo(name_or_path)
      source_path = Pathname.new(source)
      source_path = LoadModels.snapshot_download(source, python_bin: python_bin) unless source_path.exist?

      config_file = source_path.join("config.json")
      config = config_file.exist? ? JSON.parse(File.binread(config_file)) : {}
      dims_config = hf_to_dims_config(config)
      dims_config["model_type"] = "whisper"

      weights_file = [
        source_path.join("weights.npz"),
        source_path.join("model.safetensors"),
        source_path.join("weights.safetensors"),
        source_path.join("pytorch_model.bin")
      ].find(&:exist?)
      raise "No weights found in #{source_path}" if weights_file.nil?

      out_dir = Pathname.new(mlx_path)
      out_dir.mkpath

      npz_file = if weights_file.extname == ".bin"
                   extract_torch_bin(weights_file, out_dir.join("weights.npz"), python_bin: python_bin)
                 else
                   weights = MLX::Core.load(weights_file.to_s).to_a.each_with_object({}) { |(k, v), out| out[k.to_s] = v }
                   weights = cast_dtype(weights, dtype)
                   MLX::Core.savez(out_dir.join("weights.npz").to_s, **weights)
                   out_dir.join("weights.npz")
                 end

      File.binwrite(out_dir.join("config.json"), JSON.pretty_generate(dims_config) + "\n")
      puts "Saved config and weights to #{out_dir} (#{npz_file.basename})"
      out_dir
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    torch_name_or_path: "tiny",
    mlx_path: "mlx_models/tiny",
    dtype: "float16",
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby whisper/convert.rb [options]"
    opts.on("--torch-name-or-path NAME", String, "Model alias, HF repo, or local path") { |v| options[:torch_name_or_path] = v }
    opts.on("--mlx-path DIR", String, "Output directory") { |v| options[:mlx_path] = v }
    opts.on("--dtype TYPE", String, "float16|float32") { |v| options[:dtype] = v }
    opts.on("--python-bin BIN", String, "Python executable for bridge scripts") { |v| options[:python_bin] = v }
    opts.on("--list-models", "Print supported aliases") do
      puts WhisperExample::Convert.available_models.join("\n")
      exit 0
    end
  end
  parser.parse!

  WhisperExample::Convert.convert(
    options[:torch_name_or_path],
    mlx_path: options[:mlx_path],
    dtype: options[:dtype],
    python_bin: options[:python_bin]
  )
end
