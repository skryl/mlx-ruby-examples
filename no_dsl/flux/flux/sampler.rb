# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("../..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module FluxExample
  class FluxSampler
    def initialize(name, base_shift: 0.5, max_shift: 1.15)
      @base_shift = base_shift
      @max_shift = max_shift
      @schnell = name.to_s.include?("schnell")
      @cache = {}
    end

    def time_shift(x, t)
      x1 = 256.0
      x2 = 4096.0
      t1 = @base_shift
      t2 = @max_shift
      exp_mu = Math.exp((x.to_f - x1) * (t2 - t1) / (x2 - x1) + t1)

      t_arr = t.to_a
      shifted = t_arr.map do |v|
        denom = (1.0 / v.to_f) - 1.0
        exp_mu / (exp_mu + denom)
      end
      MLX::Core.array(shifted, t.dtype)
    end

    def timesteps(num_steps, image_sequence_length, start: 1.0, stop: 0.0)
      key = [num_steps.to_i, image_sequence_length.to_i, start.to_f, stop.to_f]
      return @cache[key] if @cache.key?(key)

      t = MLX::Core.linspace(start.to_f, stop.to_f, num_steps.to_i + 1)
      t = time_shift(image_sequence_length, t) unless @schnell
      out = t.to_a
      @cache[key] = out
      out
    end

    def random_timesteps(batch, length, dtype: MLX::Core.float32)
      if @schnell
        t = MLX::Core.floor(MLX::Core.random_uniform([batch], 1.0, 5.0, dtype))
        MLX::Core.divide(t, 4.0)
      else
        t = MLX::Core.random_uniform([batch], 0.0, 1.0, dtype)
        time_shift(length, t)
      end
    end

    def sample_prior(shape, dtype: MLX::Core.float32)
      MLX::Core.normal(shape).astype(dtype)
    end

    def add_noise(x, t, noise: nil)
      noise ||= MLX::Core.normal(x.shape).astype(x.dtype)
      shape = [t.shape[0]] + Array.new(x.shape.length - 1, 1)
      t = MLX::Core.reshape(t, shape)
      MLX::Core.add(
        MLX::Core.multiply(x, MLX::Core.subtract(1.0, t)),
        MLX::Core.multiply(t, noise)
      )
    end

    def step(pred, x_t, t, t_prev)
      MLX::Core.add(x_t, MLX::Core.multiply(t_prev.to_f - t.to_f, pred))
    end
  end
end
