# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module NormalizingFlowExample
  class Bijector
    def forward_and_log_det(_x)
      raise NotImplementedError
    end

    def inverse_and_log_det(_y)
      raise NotImplementedError
    end
  end

  class AffineBijector < Bijector
    def initialize(shift_and_log_scale)
      @shift_and_log_scale = shift_and_log_scale
    end

    def forward_and_log_det(x)
      shift, log_scale = MLX::Core.split(@shift_and_log_scale, 2, -1)
      y = MLX::Core.add(MLX::Core.multiply(x, MLX::Core.exp(log_scale)), shift)
      [y, log_scale]
    end

    def inverse_and_log_det(y)
      shift, log_scale = MLX::Core.split(@shift_and_log_scale, 2, -1)
      x = MLX::Core.multiply(MLX::Core.subtract(y, shift), MLX::Core.exp(MLX::Core.negative(log_scale)))
      [x, MLX::Core.negative(log_scale)]
    end
  end

  class MaskedCoupling < Bijector
    def initialize(mask, conditioner, bijector)
      @mask = mask
      @conditioner = conditioner
      @bijector = bijector
    end

    def apply_mask(x)
      x_masked = MLX::Core.where(@mask, 0.0, x)
      bijector_params = @conditioner.call(x_masked)
      y, log_det = yield(bijector_params)
      log_det = MLX::Core.where(@mask, log_det, 0.0)
      y = MLX::Core.where(@mask, y, x)
      [y, MLX::Core.sum(log_det, -1)]
    end

    def forward_and_log_det(x)
      apply_mask(x) do |params|
        @bijector.new(params).forward_and_log_det(x)
      end
    end

    def inverse_and_log_det(y)
      apply_mask(y) do |params|
        @bijector.new(params).inverse_and_log_det(y)
      end
    end
  end
end
