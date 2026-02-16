# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("../..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module FluxExample
  class LoRALinear < MLX::NN::Module
    def self.from_base(linear, r: 8, dropout: 0.0, scale: 1.0)
      output_dims, input_dims = linear.weight.shape
      lora_lin = new(
        input_dims: input_dims,
        output_dims: output_dims,
        r: r,
        dropout: dropout,
        scale: scale
      )
      lora_lin.linear = linear
      lora_lin
    end

    attr_accessor :linear

    def initialize(input_dims:, output_dims:, r: 8, dropout: 0.0, scale: 1.0, bias: false)
      super()
      self.linear = MLX::NN::Linear.new(input_dims, output_dims, bias: bias)
      self.dropout = MLX::NN::Dropout.new(dropout)
      self.scale = scale

      init_scale = 1.0 / Math.sqrt(input_dims)
      self.lora_a = MLX::Core.random_uniform([input_dims, r], -init_scale, init_scale, MLX::Core.float32)
      self.lora_b = MLX::Core.zeros([r, output_dims])
    end

    def fuse
      base = linear
      has_bias = !base.bias.nil?
      output_dims, input_dims = base.weight.shape
      fused = MLX::NN::Linear.new(input_dims, output_dims, bias: has_bias)

      lora_b_t = MLX::Core.multiply(scale, MLX::Core.transpose(lora_b, [1, 0]))
      lora_a_t = MLX::Core.transpose(lora_a, [1, 0])
      update = MLX::Core.matmul(lora_b_t, lora_a_t)
      fused.weight = MLX::Core.add(base.weight, update.astype(base.weight.dtype))
      fused.bias = base.bias if has_bias
      fused
    end

    def call(x)
      y = linear.call(x)
      z = MLX::Core.matmul(dropout.call(x), lora_a)
      z = MLX::Core.matmul(z, lora_b)
      MLX::Core.add(y, MLX::Core.multiply(scale, z).astype(x.dtype))
    end
  end
end
