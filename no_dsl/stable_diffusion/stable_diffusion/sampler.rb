# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("../..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module StableDiffusionExample
  class SimpleEulerSampler
    attr_reader :config, :max_time

    def initialize(config)
      @config = config
      @max_time = config.num_train_steps.to_f - 1.0
    end

    def sample_prior(shape, dtype: MLX::Core.float32)
      MLX::Core.normal(shape).astype(dtype)
    end

    def timesteps(num_steps, start_time: nil, dtype: MLX::Core.float32)
      start_t = start_time.nil? ? max_time : start_time.to_f
      num_steps = [num_steps.to_i, 1].max

      t_values = Array.new(num_steps + 1) do |i|
        start_t * (1.0 - i.to_f / num_steps.to_f)
      end

      Enumerator.new do |yielder|
        num_steps.times do |i|
          t = MLX::Core.array(t_values[i], dtype)
          t_prev = MLX::Core.array(t_values[i + 1], dtype)
          yielder << [t, t_prev]
        end
      end
    end

    def step(eps_pred, x_t, t, t_prev)
      dt = MLX::Core.subtract(t_prev, t)
      denom = max_time.zero? ? 1.0 : max_time
      step = MLX::Core.multiply(eps_pred, MLX::Core.divide(dt, denom))
      MLX::Core.add(x_t, step)
    end

    def add_noise(x0, t)
      denom = max_time.zero? ? 1.0 : max_time
      scale = MLX::Core.divide(t.astype(x0.dtype), denom)
      scale = MLX::Core.reshape(scale, [1, 1, 1, 1]) if scale.shape.empty?
      noise = sample_prior(x0.shape, dtype: x0.dtype)
      MLX::Core.add(x0, MLX::Core.multiply(noise, scale))
    end
  end

  class SimpleEulerAncestralSampler < SimpleEulerSampler
    def step(eps_pred, x_t, t, t_prev)
      deterministic = super
      sigma = MLX::Core.abs(MLX::Core.subtract(t, t_prev))
      denom = max_time.zero? ? 1.0 : max_time
      sigma = MLX::Core.divide(sigma, denom)
      sigma = MLX::Core.reshape(sigma, [1, 1, 1, 1]) if sigma.shape.empty?
      noise = sample_prior(x_t.shape, dtype: x_t.dtype)
      MLX::Core.add(deterministic, MLX::Core.multiply(noise, sigma))
    end
  end
end
