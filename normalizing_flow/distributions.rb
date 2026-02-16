# frozen_string_literal: true


require "mlx"

module NormalizingFlowExample
  class Normal
    def initialize(mu, sigma)
      @mu = mu
      @sigma = sigma
    end

    def sample(sample_shape, _key: nil)
      MLX::Core.add(
        MLX::Core.multiply(MLX::Core.normal(sample_shape), @sigma),
        @mu
      )
    end

    def log_prob(x)
      constant = -0.5 * Math.log(2 * Math::PI)
      centered = MLX::Core.divide(MLX::Core.subtract(x, @mu), @sigma)
      centered_sq = MLX::Core.square(centered)
      out = MLX::Core.add(constant, MLX::Core.negative(MLX::Core.log(@sigma)))
      MLX::Core.add(out, MLX::Core.multiply(-0.5, centered_sq))
    end

    def sample_and_log_prob(sample_shape, key: nil)
      x = sample(sample_shape, _key: key)
      [x, log_prob(x)]
    end
  end
end
