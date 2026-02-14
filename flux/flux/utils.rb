# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

require_relative "autoencoder"
require_relative "clip"
require_relative "model"
require_relative "t5"
require_relative "tokenizers"

module FluxExample
  SCRIPT_DIR = Pathname.new(__dir__).join("..", "python")

  class ModelSpec
    attr_reader :params, :ae_params, :ckpt_path, :ae_path, :repo_id, :repo_flow, :repo_ae

    def initialize(params:, ae_params:, ckpt_path: nil, ae_path: nil, repo_id: nil, repo_flow: nil, repo_ae: nil)
      @params = params
      @ae_params = ae_params
      @ckpt_path = ckpt_path
      @ae_path = ae_path
      @repo_id = repo_id
      @repo_flow = repo_flow
      @repo_ae = repo_ae
    end
  end

  CONFIGS = {
    "flux-dev" => ModelSpec.new(
      repo_id: "black-forest-labs/FLUX.1-dev",
      repo_flow: "flux1-dev.safetensors",
      repo_ae: "ae.safetensors",
      ckpt_path: ENV["FLUX_DEV"],
      ae_path: ENV["AE"],
      params: FluxParams.new(
        in_channels: 64,
        vec_in_dim: 768,
        context_in_dim: 1024,
        hidden_size: 512,
        mlp_ratio: 2.0,
        num_heads: 8,
        depth: 4,
        depth_single_blocks: 4,
        axes_dim: [8, 28, 28],
        theta: 10_000,
        qkv_bias: true,
        guidance_embed: true
      ),
      ae_params: AutoEncoderParams.new(
        resolution: 256,
        in_channels: 3,
        ch: 64,
        out_ch: 3,
        ch_mult: [1, 2],
        num_res_blocks: 1,
        z_channels: 16,
        scale_factor: 0.3611,
        shift_factor: 0.1159
      )
    ),
    "flux-schnell" => ModelSpec.new(
      repo_id: "black-forest-labs/FLUX.1-schnell",
      repo_flow: "flux1-schnell.safetensors",
      repo_ae: "ae.safetensors",
      ckpt_path: ENV["FLUX_SCHNELL"],
      ae_path: ENV["AE"],
      params: FluxParams.new(
        in_channels: 64,
        vec_in_dim: 768,
        context_in_dim: 1024,
        hidden_size: 512,
        mlp_ratio: 2.0,
        num_heads: 8,
        depth: 4,
        depth_single_blocks: 4,
        axes_dim: [8, 28, 28],
        theta: 10_000,
        qkv_bias: true,
        guidance_embed: false
      ),
      ae_params: AutoEncoderParams.new(
        resolution: 256,
        in_channels: 3,
        ch: 64,
        out_ch: 3,
        ch_mult: [1, 2],
        num_res_blocks: 1,
        z_channels: 16,
        scale_factor: 0.3611,
        shift_factor: 0.1159
      )
    )
  }.freeze

  module_function

  def resolve_model_name(name)
    model_name = name.to_s
    return model_name if CONFIGS.key?(model_name)

    if model_name == "dev"
      "flux-dev"
    elsif model_name == "schnell"
      "flux-schnell"
    elsif model_name.start_with?("flux-")
      model_name
    else
      "flux-#{model_name}"
    end
  end

  def hf_download(repo_id, filename, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    script = SCRIPT_DIR.join("hf_download.py").to_s
    stdout, stderr, status = Open3.capture3(python_bin, script, repo_id.to_s, filename.to_s)
    unless status.success?
      raise RuntimeError, "huggingface download failed (#{repo_id}/#{filename}): #{stderr}"
    end

    stdout.strip
  end

  def maybe_load_weights(model, path)
    return model if path.nil? || path.to_s.empty? || !File.exist?(path)

    weights = MLX::Core.load(path)
    if model.respond_to?(:sanitize)
      weights = model.sanitize(weights)
    end
    model.load_weights(weights.to_a, strict: false)
    model
  end

  def load_flow_model(name, hf_download_enabled: true, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    model_name = resolve_model_name(name)
    spec = CONFIGS.fetch(model_name)
    ckpt_path = spec.ckpt_path

    if ckpt_path.nil? && hf_download_enabled && !spec.repo_id.nil? && !spec.repo_flow.nil?
      begin
        ckpt_path = hf_download(spec.repo_id, spec.repo_flow, python_bin: python_bin)
      rescue StandardError
        ckpt_path = nil
      end
    end

    model = Flux.new(spec.params)
    maybe_load_weights(model, ckpt_path)
  end

  def load_ae(name, hf_download_enabled: true, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    model_name = resolve_model_name(name)
    spec = CONFIGS.fetch(model_name)
    ckpt_path = spec.ae_path

    if ckpt_path.nil? && hf_download_enabled && !spec.repo_id.nil? && !spec.repo_ae.nil?
      begin
        ckpt_path = hf_download(spec.repo_id, spec.repo_ae, python_bin: python_bin)
      rescue StandardError
        ckpt_path = nil
      end
    end

    ae = AutoEncoder.new(spec.ae_params)
    maybe_load_weights(ae, ckpt_path)
  end

  def load_clip(name)
    _ = resolve_model_name(name)
    config = CLIPTextModelConfig.new(
      num_layers: 6,
      model_dims: 768,
      num_heads: 12,
      max_length: 77,
      vocab_size: 49_408,
      hidden_act: "quick_gelu"
    )
    CLIPTextModel.new(config)
  end

  def load_t5(name)
    _ = resolve_model_name(name)
    config = T5Config.new(
      vocab_size: 32_128,
      num_layers: 6,
      num_heads: 8,
      relative_attention_num_buckets: 32,
      d_kv: 64,
      d_model: 1024,
      feed_forward_proj: "gelu",
      tie_word_embeddings: false,
      d_ff: 2048,
      num_decoder_layers: 6
    )
    T5Encoder.new(config)
  end

  def load_clip_tokenizer(name)
    _ = resolve_model_name(name)
    CLIPTokenizer.new({}, {}, max_length: 77)
  end

  def load_t5_tokenizer(name, _pad: true)
    model_name = resolve_model_name(name)
    length = model_name.include?("schnell") ? 256 : 512
    T5Tokenizer.new(nil, max_length: length)
  end

  def save_config(config, config_path)
    sorted = config.keys.sort.each_with_object({}) do |key, out|
      out[key] = config[key]
    end
    File.binwrite(config_path.to_s, JSON.pretty_generate(sorted))
  end
end
