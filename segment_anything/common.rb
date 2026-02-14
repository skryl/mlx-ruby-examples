# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module SegmentAnything
  class MLPBlock < MLX::NN::Module
    def initialize(embedding_dim:, mlp_dim:)
      super()
      self.lin1 = MLX::NN::Linear.new(embedding_dim, mlp_dim)
      self.lin2 = MLX::NN::Linear.new(mlp_dim, embedding_dim)
      self.act = MLX::NN::GELU.new
    end

    def call(x)
      lin2.call(act.call(lin1.call(x)))
    end
  end

  class LayerNorm2d < MLX::NN::Module
    def initialize(num_channels, eps: 1e-6)
      super()
      self.weight = MLX::Core.ones([num_channels])
      self.bias = MLX::Core.zeros([num_channels])
      @eps = eps
    end

    def call(x)
      u = MLX::Core.expand_dims(MLX::Core.mean(x, 3), 3)
      centered = MLX::Core.subtract(x, u)
      s = MLX::Core.expand_dims(MLX::Core.mean(MLX::Core.square(centered), 3), 3)
      y = MLX::Core.divide(centered, MLX::Core.sqrt(MLX::Core.add(s, @eps)))
      MLX::Core.add(MLX::Core.multiply(weight, y), bias)
    end
  end
end
