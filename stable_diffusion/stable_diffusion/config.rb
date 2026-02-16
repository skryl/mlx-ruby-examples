# frozen_string_literal: true

module StableDiffusionExample
  class AutoencoderConfig
    include MLX::DSL::ConfigSchema

    field :in_channels, Integer, default: 3
    field :out_channels, Integer, default: 3
    field :latent_channels_out, Integer, default: 8
    field :latent_channels_in, Integer, default: 4
    field :block_out_channels, Array, default: [128, 256, 512, 512]
    field :layers_per_block, Integer, default: 2
    field :norm_num_groups, Integer, default: 32
    field :scaling_factor, [Integer, Float], default: 0.18215
  end

  class CLIPTextModelConfig
    include MLX::DSL::ConfigSchema

    field :num_layers, Integer, default: 23
    field :model_dims, Integer, default: 1024
    field :num_heads, Integer, default: 16
    field :max_length, Integer, default: 77
    field :vocab_size, Integer, default: 49_408
    field :projection_dim, [Integer, NilClass], default: nil
    field :hidden_act, String, default: "quick_gelu"
  end

  class UNetConfig
    include MLX::DSL::ConfigSchema

    field :in_channels, Integer, default: 4
    field :out_channels, Integer, default: 4
    field :conv_in_kernel, Integer, default: 3
    field :conv_out_kernel, Integer, default: 3
    field :block_out_channels, Array, default: [320, 640, 1280, 1280]
    field :layers_per_block, Array, default: [2, 2, 2, 2]
    field :mid_block_layers, Integer, default: 2
    field :transformer_layers_per_block, Array, default: [1, 1, 1, 1]
    field :num_attention_heads, Array, default: [5, 10, 20, 20]
    field :cross_attention_dim, Array, default: [1024, 1024, 1024, 1024]
    field :norm_num_groups, Integer, default: 32
    field :down_block_types, Array, default: ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]
    field :up_block_types, Array, default: ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"]
    field :addition_embed_type, [String, NilClass], default: nil
    field :addition_time_embed_dim, [Integer, NilClass], default: nil
    field :projection_class_embeddings_input_dim, [Integer, NilClass], default: nil
  end

  class DiffusionConfig
    include MLX::DSL::ConfigSchema

    field :beta_schedule, String, default: "scaled_linear"
    field :beta_start, [Integer, Float], default: 0.00085
    field :beta_end, [Integer, Float], default: 0.012
    field :num_train_steps, Integer, default: 1000
  end
end
