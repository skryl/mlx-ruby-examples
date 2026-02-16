# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module GcnExample
  module Datasets
    module_function

    PREPARE_SCRIPT = Pathname.new(__dir__).join("python", "prepare_cora.py").to_s

    def train_val_test_mask(num_nodes:)
      if num_nodes >= 1500
        train_idx = (0...140).to_a
        val_idx = (200...500).to_a
        test_idx = (500...1500).to_a
      else
        train_end = (num_nodes * 0.2).to_i
        val_end = (num_nodes * 0.4).to_i
        train_idx = (0...train_end).to_a
        val_idx = (train_end...val_end).to_a
        test_idx = (val_end...num_nodes).to_a
      end

      [
        MLX::Core.array(train_idx, MLX::Core.int32),
        MLX::Core.array(val_idx, MLX::Core.int32),
        MLX::Core.array(test_idx, MLX::Core.int32)
      ]
    end

    def load_data(
      synthetic: false,
      data_root: nil,
      synthetic_nodes: 512,
      feature_dim: 64,
      classes: 7,
      edge_prob: 0.01,
      seed: 0,
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
      return synthetic_data(
        nodes: synthetic_nodes,
        feature_dim: feature_dim,
        classes: classes,
        edge_prob: edge_prob,
        seed: seed
      ) if synthetic

      root_dir = data_root.nil? ? Pathname.new(__dir__).join("data") : Pathname.new(data_root.to_s)
      root_dir = root_dir.expand_path
      out_file = root_dir.join("cora_processed.npz")
      ensure_prepared(out_file, root_dir, python_bin)
      load_npz(out_file)
    end

    def ensure_prepared(out_file, root_dir, python_bin)
      return if out_file.exist?

      root_dir.mkpath
      stdout, stderr, status = Open3.capture3(
        python_bin,
        PREPARE_SCRIPT,
        out_file.to_s,
        root_dir.to_s
      )
      return if status.success? && out_file.exist?

      raise RuntimeError, "Failed to prepare Cora dataset: #{stderr}\n#{stdout}"
    end

    def load_npz(path)
      data = MLX::Core.load(path.to_s).to_a.to_h.transform_keys(&:to_s)
      features = data.fetch("features").astype(MLX::Core.float32)
      labels = data.fetch("labels").astype(MLX::Core.int32)
      adjacency = data.fetch("adjacency").astype(MLX::Core.float32)
      [features, labels, adjacency]
    end

    def synthetic_data(nodes:, feature_dim:, classes:, edge_prob:, seed:)
      rng = Random.new(seed)

      features = MLX::Core.normal([nodes, feature_dim]).astype(MLX::Core.float32)
      labels = Array.new(nodes) { rng.rand(0...classes) }

      adj = Array.new(nodes) { Array.new(nodes, 0.0) }
      nodes.times { |i| adj[i][i] = 1.0 }
      nodes.times do |i|
        ((i + 1)...nodes).each do |j|
          next unless rng.rand < edge_prob

          adj[i][j] = 1.0
          adj[j][i] = 1.0
        end
      end

      degree = Array.new(nodes, 0.0)
      nodes.times do |i|
        degree[i] = adj[i].sum
      end

      nodes.times do |i|
        nodes.times do |j|
          next if adj[i][j].zero?

          denom = Math.sqrt(degree[i] * degree[j])
          adj[i][j] = denom.zero? ? 0.0 : (adj[i][j] / denom)
        end
      end

      [
        features,
        MLX::Core.array(labels, MLX::Core.int32),
        MLX::Core.array(adj, MLX::Core.float32)
      ]
    end
  end
end
