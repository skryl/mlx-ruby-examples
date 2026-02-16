# frozen_string_literal: true

require_relative "bijectors"
require_relative "distributions"

module NormalizingFlowExample
  class MLP < MLX::NN::Module
    def initialize(n_layers, d_in, d_hidden, d_out)
      super()
      layer_sizes = [d_in] + Array.new(n_layers, d_hidden) + [d_out]
      self.layers = layer_sizes.each_cons(2).map { |idim, odim| MLX::NN::Linear.new(idim, odim) }
    end

    def call(x)
      hidden = x
      layers[0...-1].each do |layer|
        hidden = MLX::NN.gelu(layer.call(hidden))
      end
      layers[-1].call(hidden)
    end
  end

  class RealNVP < MLX::NN::Module
    def initialize(n_transforms, d_params, d_hidden, n_layers)
      super()
      @mask_list = Array.new(n_transforms) do |i|
        mask = (0...d_params).map { |j| (j % 2) == (i % 2) }
        MLX::Core.array(mask, MLX::Core.bool_)
      end
      self.conditioner_list = Array.new(n_transforms) do
        MLP.new(n_layers, d_params, d_hidden, 2 * d_params)
      end
      @base_dist = Normal.new(
        MLX::Core.zeros([d_params]),
        MLX::Core.ones([d_params])
      )
    end

    def log_prob(x)
      log_prob = MLX::Core.zeros([x.shape[0]])
      @mask_list.reverse.each_with_index do |mask, idx|
        conditioner = conditioner_list[conditioner_list.length - 1 - idx]
        coupling = MaskedCoupling.new(mask, conditioner, AffineBijector)
        x, ldj = coupling.inverse_and_log_det(x)
        log_prob = MLX::Core.add(log_prob, ldj)
      end
      base = MLX::Core.sum(@base_dist.log_prob(x), -1)
      MLX::Core.add(log_prob, base)
    end

    def sample(sample_shape, key: nil, n_transforms: nil)
      x = @base_dist.sample(sample_shape, _key: key)
      limit = n_transforms.nil? ? @mask_list.length : n_transforms
      limit = [[limit, 0].max, @mask_list.length].min
      limit.times do |i|
        coupling = MaskedCoupling.new(@mask_list[i], conditioner_list[i], AffineBijector)
        x, = coupling.forward_and_log_det(x)
      end
      x
    end

    def call(x)
      log_prob(x)
    end
  end
end
