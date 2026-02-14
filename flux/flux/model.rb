# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("../..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

require_relative "layers"

module FluxExample
  class FluxParams
    attr_reader :in_channels,
                :vec_in_dim,
                :context_in_dim,
                :hidden_size,
                :mlp_ratio,
                :num_heads,
                :depth,
                :depth_single_blocks,
                :axes_dim,
                :theta,
                :qkv_bias,
                :guidance_embed

    def initialize(
      in_channels:,
      vec_in_dim:,
      context_in_dim:,
      hidden_size:,
      mlp_ratio:,
      num_heads:,
      depth:,
      depth_single_blocks:,
      axes_dim:,
      theta:,
      qkv_bias:,
      guidance_embed:
    )
      @in_channels = in_channels
      @vec_in_dim = vec_in_dim
      @context_in_dim = context_in_dim
      @hidden_size = hidden_size
      @mlp_ratio = mlp_ratio
      @num_heads = num_heads
      @depth = depth
      @depth_single_blocks = depth_single_blocks
      @axes_dim = axes_dim
      @theta = theta
      @qkv_bias = qkv_bias
      @guidance_embed = guidance_embed
    end
  end

  class Flux < MLX::NN::Module
    attr_reader :params, :in_channels, :out_channels, :hidden_size

    def initialize(params)
      super()
      @params = params
      @in_channels = params.in_channels
      @out_channels = @in_channels

      if (params.hidden_size % params.num_heads) != 0
        raise ArgumentError, "Hidden size #{params.hidden_size} must be divisible by num_heads #{params.num_heads}"
      end

      pe_dim = params.hidden_size / params.num_heads
      if params.axes_dim.sum != pe_dim
        raise ArgumentError, "axes_dim sum #{params.axes_dim.sum} must match positional dim #{pe_dim}"
      end

      @hidden_size = params.hidden_size
      self.pe_embedder = EmbedND.new(dim: pe_dim, theta: params.theta, axes_dim: params.axes_dim)
      self.img_in = MLX::NN::Linear.new(@in_channels, @hidden_size, bias: true)
      self.time_in = MLPEmbedder.new(in_dim: 256, hidden_dim: @hidden_size)
      self.vector_in = MLPEmbedder.new(in_dim: params.vec_in_dim, hidden_dim: @hidden_size)
      self.guidance_in = params.guidance_embed ? MLPEmbedder.new(in_dim: 256, hidden_dim: @hidden_size) : nil
      self.txt_in = MLX::NN::Linear.new(params.context_in_dim, @hidden_size)

      self.double_blocks = Array.new(params.depth) do
        DoubleStreamBlock.new(
          @hidden_size,
          params.num_heads,
          mlp_ratio: params.mlp_ratio,
          qkv_bias: params.qkv_bias
        )
      end

      self.single_blocks = Array.new(params.depth_single_blocks) do
        SingleStreamBlock.new(@hidden_size, params.num_heads, mlp_ratio: params.mlp_ratio)
      end

      self.final_layer = LastLayer.new(@hidden_size, 1, @out_channels)
    end

    def sanitize(weights)
      new_weights = {}
      weights.each do |k, w|
        key = k.to_s
        key = key[22..] if key.start_with?("model.diffusion_model.")
        key = "#{key[0...-6]}.weight" if key.end_with?(".scale")
        %w[img_mlp txt_mlp adaLN_modulation].each do |seq|
          needle = ".#{seq}."
          if key.include?(needle)
            key = key.gsub(needle, ".#{seq}.layers.")
            break
          end
        end
        new_weights[key] = w
      end
      new_weights
    end

    def shard(_group = nil)
      # no-op in this Ruby conversion
    end

    def call(img:, img_ids:, txt:, txt_ids:, timesteps:, y:, guidance: nil)
      unless img.shape.length == 3 && txt.shape.length == 3
        raise ArgumentError, "Input img and txt tensors must have 3 dimensions"
      end

      img = img_in.call(img)
      vec = time_in.call(Layers.timestep_embedding(timesteps, 256))
      if params.guidance_embed
        if guidance.nil?
          raise ArgumentError, "guidance must be provided for guidance distilled model"
        end
        vec = MLX::Core.add(vec, guidance_in.call(Layers.timestep_embedding(guidance, 256)))
      end
      vec = MLX::Core.add(vec, vector_in.call(y))
      txt = txt_in.call(txt)

      ids = MLX::Core.concatenate([txt_ids, img_ids], 1)
      pe = pe_embedder.call(ids).astype(img.dtype)

      double_blocks.each do |block|
        img, txt = block.call(img: img, txt: txt, vec: vec, pe: pe)
      end

      joined = MLX::Core.concatenate([txt, img], 1)
      single_blocks.each do |block|
        joined = block.call(joined, vec: vec, pe: pe)
      end

      txt_len = txt.shape[1]
      joined = MLX::Core.slice(
        joined,
        [0, txt_len, 0],
        [joined.shape[0], joined.shape[1], joined.shape[2]]
      )

      final_layer.call(joined, vec)
    end
  end
end
