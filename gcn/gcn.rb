# frozen_string_literal: true

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

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

  class GCN < MLX::NN::Module
    def initialize(x_dim:, h_dim:, out_dim:, nb_layers: 2, dropout: 0.5, bias: true)
      super()
      layer_sizes = [x_dim] + Array.new(nb_layers, h_dim) + [out_dim]
      self.gcn_layers = layer_sizes.each_cons(2).map do |in_dim, out_dim_curr|
        GCNLayer.new(in_dim, out_dim_curr, bias: bias)
      end
      self.dropout = MLX::NN::Dropout.new(dropout)
    end

    def call(x, adj)
      hidden = x
      gcn_layers[0...-1].each do |layer|
        hidden = MLX::NN.relu(layer.call(hidden, adj))
        hidden = dropout.call(hidden)
      end
      gcn_layers[-1].call(hidden, adj)
    end
  end
end
