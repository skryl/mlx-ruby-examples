# frozen_string_literal: true

require_relative "clip"
require_relative "config"
require_relative "tokenizer"
require_relative "unet"
require_relative "vae"

module StableDiffusionExample
  DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1-base"
  MODELS = {
    "stabilityai/sdxl-turbo" => :sdxl,
    "stabilityai/stable-diffusion-2-1-base" => :sd
  }.freeze

  module_function

  def check_key(key, part)
    return if MODELS.key?(key)

    raise ArgumentError, "[#{part}] '#{key}' model not found, choose one of #{MODELS.keys.join(', ')}"
  end

  def load_unet(key = DEFAULT_MODEL, _float16 = false)
    check_key(key, "load_unet")

    dims = MODELS[key] == :sdxl ? 128 : 96
    cross = MODELS[key] == :sdxl ? 256 : 96
    config = UNetConfig.new(
      in_channels: 4,
      out_channels: 4,
      block_out_channels: [dims, dims * 2],
      layers_per_block: [2, 2],
      transformer_layers_per_block: [1, 1],
      num_attention_heads: [4, 4],
      cross_attention_dim: [cross, cross],
      down_block_types: ["CrossAttnDownBlock2D", "DownBlock2D"],
      up_block_types: ["UpBlock2D", "CrossAttnUpBlock2D"]
    )
    UNetModel.new(config)
  end

  def load_text_encoder(
    key = DEFAULT_MODEL,
    _float16 = false,
    model_key: "text_encoder",
    config_key: nil
  )
    check_key(key, "load_text_encoder")
    _ = config_key

    dims = MODELS[key] == :sdxl ? 128 : 96
    with_projection = model_key == "text_encoder_2"

    CLIPTextModel.new(
      CLIPTextModelConfig.new(
        num_layers: 4,
        model_dims: dims,
        num_heads: 4,
        max_length: 77,
        vocab_size: 50_000,
        projection_dim: with_projection ? dims : nil,
        hidden_act: "quick_gelu"
      )
    )
  end

  def load_autoencoder(key = DEFAULT_MODEL, _float16 = false)
    check_key(key, "load_autoencoder")

    Autoencoder.new(
      AutoencoderConfig.new(
        in_channels: 3,
        out_channels: 3,
        latent_channels_out: 8,
        latent_channels_in: 4,
        block_out_channels: [64, 128],
        layers_per_block: 2,
        norm_num_groups: 8,
        scaling_factor: 0.18215
      )
    )
  end

  def load_diffusion_config(key = DEFAULT_MODEL)
    check_key(key, "load_diffusion_config")

    DiffusionConfig.new(
      beta_start: 0.00085,
      beta_end: 0.012,
      beta_schedule: "scaled_linear",
      num_train_steps: 100
    )
  end

  def load_tokenizer(
    key = DEFAULT_MODEL,
    vocab_key: "tokenizer_vocab",
    merges_key: "tokenizer_merges"
  )
    check_key(key, "load_tokenizer")
    _ = vocab_key
    _ = merges_key

    Tokenizer.new(nil, nil, max_length: 77)
  end
end
