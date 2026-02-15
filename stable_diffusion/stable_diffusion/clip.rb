# frozen_string_literal: true


require "mlx"

require_relative "config"

module StableDiffusionExample
  class CLIPOutput
    attr_reader :last_hidden_state, :hidden_states, :pooled_output

    def initialize(last_hidden_state:, hidden_states:, pooled_output:)
      @last_hidden_state = last_hidden_state
      @hidden_states = hidden_states
      @pooled_output = pooled_output
    end
  end

  class CLIPTextModel < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config

      self.token_embedding = MLX::NN::Embedding.new(config.vocab_size, config.model_dims)
      layer_count = [config.num_layers, 4].min
      self.layers = Array.new(layer_count) { MLX::NN::Linear.new(config.model_dims, config.model_dims) }
      self.final_norm = MLX::NN::LayerNorm.new(config.model_dims)
      self.text_projection = if config.projection_dim.nil?
        nil
      else
        MLX::NN::Linear.new(config.model_dims, config.projection_dim, bias: false)
      end
    end

    def call(tokens)
      x = token_embedding.call(tokens)
      hidden_states = []
      layers.each do |layer|
        x = quick_gelu(layer.call(x))
        hidden_states << x
      end
      last_hidden = final_norm.call(x)
      pooled = MLX::Core.mean(last_hidden, 1)
      pooled = text_projection.call(pooled) unless text_projection.nil?

      CLIPOutput.new(
        last_hidden_state: last_hidden,
        hidden_states: hidden_states,
        pooled_output: pooled
      )
    end

    private

    def quick_gelu(x)
      MLX::Core.multiply(x, MLX::Core.sigmoid(MLX::Core.multiply(1.702, x)))
    end
  end
end
