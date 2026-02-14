# frozen_string_literal: true

require "json"
require "optparse"
require "open3"
require "pathname"

require_relative "encodec"

module EncodecExample
  UPLOAD_SCRIPT = Pathname.new(__dir__).join("python", "upload_to_hub.py").to_s

  module Convert
    module_function

    def fetch_from_hub(repo_id, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      EncodecExample.snapshot_download(repo_id, python_bin: python_bin)
    end

    def cast_dtype(weights, dtype)
      return weights if dtype.nil?

      type = case dtype
             when "float32" then MLX::Core.float32
             when "float16" then MLX::Core.float16
             when "bfloat16" then MLX::Core.bfloat16
             else
               raise ArgumentError, "Unsupported dtype '#{dtype}'"
             end

      weights.transform_values { |v| v.astype(type) }
    end

    def save_weights(save_dir, weights)
      save_dir = Pathname.new(save_dir)
      save_dir.mkpath
      MLX::Core.savez(save_dir.join("weights.npz").to_s, **weights)
    end

    def save_config(config, path)
      clean = config.reject { |k, _| k == "_name_or_path" }
      sorted = clean.keys.sort.each_with_object({}) { |k, out| out[k] = clean[k] }
      File.binwrite(path, JSON.pretty_generate(sorted) + "\n")
    end

    def upload_to_hub(path, repo_id, source_repo, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      stdout, stderr, status = Open3.capture3(
        python_bin,
        UPLOAD_SCRIPT,
        path.to_s,
        repo_id.to_s,
        source_repo.to_s
      )
      raise "Upload failed: #{stderr}" unless status.success?

      puts stdout
    end

    def convert(model:, dtype:, output:, upload:, upload_repo:, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      hf_repo = "facebook/encodec_#{model}"
      path = fetch_from_hub(hf_repo, python_bin: python_bin)

      weight_file = Dir.glob(path.join("*.safetensors").to_s).first || Dir.glob(path.join("*.npz").to_s).first
      raise "No weights found under #{path}" if weight_file.nil?

      weights = MLX::Core.load(weight_file).to_a.each_with_object({}) { |(k, v), out| out[k.to_s] = v }
      weights = cast_dtype(weights, dtype)

      out_dir = Pathname.new(output)
      out_dir.mkpath
      save_weights(out_dir, weights)

      config_path = path.join("config.json")
      if config_path.exist?
        config = JSON.parse(File.binread(config_path))
        save_config(config, out_dir.join("config.json"))
      end

      if upload
        target_repo = upload_repo || "mlx-community/encodec-#{model}-#{dtype || 'float32'}"
        upload_to_hub(out_dir, target_repo, hf_repo, python_bin: python_bin)
      end

      puts "Saved converted checkpoint to #{out_dir}"
      out_dir
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model: "48khz",
    dtype: "float32",
    output: "mlx_models/encodec",
    upload: false,
    upload_repo: nil,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby encodec/convert.rb [options]"
    opts.on("--model NAME", String, "Model variant: 24khz, 32khz, 48khz") { |v| options[:model] = v }
    opts.on("--dtype TYPE", String, "Output dtype: float32, float16, bfloat16") { |v| options[:dtype] = v }
    opts.on("--output DIR", String, "Output directory") { |v| options[:output] = v }
    opts.on("--upload", "Upload converted files to Hugging Face") { options[:upload] = true }
    opts.on("--upload-repo REPO", String, "Target upload repo id") { |v| options[:upload_repo] = v }
    opts.on("--python-bin BIN", String, "Python executable for bridge scripts") { |v| options[:python_bin] = v }
  end
  parser.parse!

  unless ["24khz", "32khz", "48khz"].include?(options[:model])
    raise ArgumentError, "model must be one of: 24khz, 32khz, 48khz"
  end

  EncodecExample::Convert.convert(
    model: options[:model],
    dtype: options[:dtype],
    output: options[:output],
    upload: options[:upload],
    upload_repo: options[:upload_repo],
    python_bin: options[:python_bin]
  )
end
