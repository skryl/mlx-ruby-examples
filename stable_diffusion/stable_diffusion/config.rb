# frozen_string_literal: true

module StableDiffusionExample
  class AutoencoderConfig
    attr_reader :in_channels, :out_channels, :latent_channels_out, :latent_channels_in,
                :block_out_channels, :layers_per_block, :norm_num_groups, :scaling_factor

    def initialize(
      in_channels: 3,
      out_channels: 3,
      latent_channels_out: 8,
      latent_channels_in: 4,
      block_out_channels: [128, 256, 512, 512],
      layers_per_block: 2,
      norm_num_groups: 32,
      scaling_factor: 0.18215
    )
      @in_channels = in_channels
      @out_channels = out_channels
      @latent_channels_out = latent_channels_out
      @latent_channels_in = latent_channels_in
      @block_out_channels = block_out_channels
      @layers_per_block = layers_per_block
      @norm_num_groups = norm_num_groups
      @scaling_factor = scaling_factor
    end
  end

  class CLIPTextModelConfig
    attr_reader :num_layers, :model_dims, :num_heads, :max_length, :vocab_size, :projection_dim, :hidden_act

    def initialize(
      num_layers: 23,
      model_dims: 1024,
      num_heads: 16,
      max_length: 77,
      vocab_size: 49_408,
      projection_dim: nil,
      hidden_act: "quick_gelu"
    )
      @num_layers = num_layers
      @model_dims = model_dims
      @num_heads = num_heads
      @max_length = max_length
      @vocab_size = vocab_size
      @projection_dim = projection_dim
      @hidden_act = hidden_act
    end
  end

  class UNetConfig
    attr_reader :in_channels,
                :out_channels,
                :conv_in_kernel,
                :conv_out_kernel,
                :block_out_channels,
                :layers_per_block,
                :mid_block_layers,
                :transformer_layers_per_block,
                :num_attention_heads,
                :cross_attention_dim,
                :norm_num_groups,
                :down_block_types,
                :up_block_types,
                :addition_embed_type,
                :addition_time_embed_dim,
                :projection_class_embeddings_input_dim

    def initialize(
      in_channels: 4,
      out_channels: 4,
      conv_in_kernel: 3,
      conv_out_kernel: 3,
      block_out_channels: [320, 640, 1280, 1280],
      layers_per_block: [2, 2, 2, 2],
      mid_block_layers: 2,
      transformer_layers_per_block: [1, 1, 1, 1],
      num_attention_heads: [5, 10, 20, 20],
      cross_attention_dim: [1024, 1024, 1024, 1024],
      norm_num_groups: 32,
      down_block_types: ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
      up_block_types: ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
      addition_embed_type: nil,
      addition_time_embed_dim: nil,
      projection_class_embeddings_input_dim: nil
    )
      @in_channels = in_channels
      @out_channels = out_channels
      @conv_in_kernel = conv_in_kernel
      @conv_out_kernel = conv_out_kernel
      @block_out_channels = block_out_channels
      @layers_per_block = layers_per_block
      @mid_block_layers = mid_block_layers
      @transformer_layers_per_block = transformer_layers_per_block
      @num_attention_heads = num_attention_heads
      @cross_attention_dim = cross_attention_dim
      @norm_num_groups = norm_num_groups
      @down_block_types = down_block_types
      @up_block_types = up_block_types
      @addition_embed_type = addition_embed_type
      @addition_time_embed_dim = addition_time_embed_dim
      @projection_class_embeddings_input_dim = projection_class_embeddings_input_dim
    end
  end

  class DiffusionConfig
    attr_reader :beta_schedule, :beta_start, :beta_end, :num_train_steps

    def initialize(
      beta_schedule: "scaled_linear",
      beta_start: 0.00085,
      beta_end: 0.012,
      num_train_steps: 1000
    )
      @beta_schedule = beta_schedule
      @beta_start = beta_start
      @beta_end = beta_end
      @num_train_steps = num_train_steps
    end
  end
end
