# frozen_string_literal: true


require "mlx"
require "mlx/dsl"

module GcnExample
  class GCNLayer < MLX::NN::Module
    def initialize(in_features, out_features, bias: true)
      super()
      self.linear = MLX::NN::Linear.new(in_features, out_features, bias: bias)
    end

    def call(x, adj)
      MLX::Core.matmul(adj, linear.call(x))
    end
  end

  class GCN < MLX::DSL::Model
    option :x_dim
    option :h_dim
    option :out_dim
    option :nb_layers, default: 2
    option :dropout, default: 0.5
    option :bias, default: true

    layer :gcn_stack do
      layer_sizes = [x_dim] + Array.new(nb_layers, h_dim) + [out_dim]
      MLX::NN::Sequential.new(
        *layer_sizes.each_cons(2).map { |in_dim, out_dim_curr| GCNLayer.new(in_dim, out_dim_curr, bias: bias) }
      )
    end

    layer :dropout_layer, MLX::NN::Dropout, -> { dropout }

    def gcn_layers
      gcn_stack.layers
    end

    def call(x, adj)
      hidden = x
      gcn_layers[0...-1].each do |layer|
        hidden = MLX::NN.relu(layer.call(hidden, adj))
        hidden = dropout_layer.call(hidden)
      end
      gcn_layers[-1].call(hidden, adj)
    end
  end
end
