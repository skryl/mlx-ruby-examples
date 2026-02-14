# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("../..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

require_relative "config"

module StableDiffusionExample
  class UNetModel < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config

      hidden = config.block_out_channels.first
      self.cond_proj = MLX::NN::Linear.new(config.cross_attention_dim.first, config.in_channels, bias: false)
      self.time_proj = MLX::NN::Linear.new(1, config.in_channels, bias: true)
      self.conv_in = MLX::NN::Linear.new(config.in_channels, hidden)
      self.mid = MLX::NN::Linear.new(hidden, hidden)
      self.conv_out = MLX::NN::Linear.new(hidden, config.out_channels)
    end

    def call(x_t, t, encoder_x:, text_time: nil)
      batch = x_t.shape[0]

      cond = MLX::Core.mean(encoder_x, 1)
      cond = cond_proj.call(cond)
      cond = MLX::Core.reshape(cond, [batch, 1, 1, cond.shape[1]])

      t = MLX::Core.reshape(t, [batch, 1])
      t_embed = time_proj.call(t)
      t_embed = MLX::Core.reshape(t_embed, [batch, 1, 1, t_embed.shape[1]])

      h = MLX::Core.add(x_t, cond)
      h = MLX::Core.add(h, t_embed)

      h = MLX::NN.silu(conv_in.call(h))
      h = MLX::NN.silu(mid.call(h))

      _ = text_time

      conv_out.call(h)
    end
  end
end
