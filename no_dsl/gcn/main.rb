# frozen_string_literal: true

require "optparse"
require "time"

require_relative "datasets"
require_relative "gcn"

module GcnExample
  module Train
    module_function

    def select_rows(x, indices)
      MLX::Core.take(x, indices, 0)
    end

    def loss_fn(y_hat, y, weight_decay: 0.0, parameters: nil)
      loss = MLX::Core.mean(MLX::NN::Losses.cross_entropy(y_hat, y))
      return loss if weight_decay.to_f.zero?
      raise ArgumentError, "Model parameters missing for L2 regularization." if parameters.nil?

      l2_reg = nil
      MLX::Utils.tree_flatten(parameters).each do |_name, value|
        term = MLX::Core.sum(MLX::Core.square(value))
        l2_reg = l2_reg.nil? ? term : MLX::Core.add(l2_reg, term)
      end
      l2_reg = MLX::Core.sqrt(l2_reg)
      MLX::Core.add(loss, MLX::Core.multiply(weight_decay, l2_reg))
    end

    def eval_fn(logits, labels)
      preds = MLX::Core.argmax(logits, 1)
      MLX::Core.mean(MLX::Core.equal(preds, labels))
    end

    def forward_loss(gcn, x, adj, y, train_mask, weight_decay)
      y_hat = gcn.call(x, adj)
      train_logits = select_rows(y_hat, train_mask)
      train_labels = select_rows(y, train_mask)
      loss_fn(
        train_logits,
        train_labels,
        weight_decay: weight_decay,
        parameters: gcn.parameters
      )
    end

    def run(options)
      MLX::Core.set_default_device(MLX::Core.cpu) if options[:cpu]
      MLX::Core.random_seed(options[:seed])

      x, y, adj = Datasets.load_data(
        synthetic: options[:synthetic],
        data_root: options[:data_root],
        synthetic_nodes: options[:synthetic_nodes],
        feature_dim: options[:synthetic_feature_dim],
        classes: options[:nb_classes],
        edge_prob: options[:synthetic_edge_prob],
        seed: options[:seed],
        python_bin: options[:python_bin]
      )
      train_mask, val_mask, test_mask = Datasets.train_val_test_mask(num_nodes: x.shape[0])

      gcn = GCN.new(
        x_dim: x.shape[-1],
        h_dim: options[:hidden_dim],
        out_dim: options[:nb_classes],
        nb_layers: options[:nb_layers],
        dropout: options[:dropout],
        bias: options[:bias]
      )
      MLX::Core.eval(gcn.parameters)

      optimizer = MLX::Optimizers::Adam.new(learning_rate: options[:lr])
      loss_and_grad_fn = MLX::NN.value_and_grad(
        gcn,
        lambda do |features, graph_adj, labels, mask|
          forward_loss(gcn, features, graph_adj, labels, mask, options[:weight_decay])
        end
      )

      best_val_loss = Float::INFINITY
      no_improve = 0

      options[:epochs].times do |epoch|
        tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        loss, grads = loss_and_grad_fn.call(x, adj, y, train_mask)
        optimizer.update(gcn, grads)
        y_hat = gcn.call(x, adj)

        val_loss = loss_fn(select_rows(y_hat, val_mask), select_rows(y, val_mask))
        val_acc = eval_fn(select_rows(y_hat, val_mask), select_rows(y, val_mask))
        MLX::Core.eval(loss, val_loss, val_acc, gcn.parameters, optimizer.state)
        toc = Process.clock_gettime(Process::CLOCK_MONOTONIC)

        val_loss_scalar = val_loss.item.to_f
        if val_loss_scalar < best_val_loss
          best_val_loss = val_loss_scalar
          no_improve = 0
        else
          no_improve += 1
          break if no_improve >= options[:patience]
        end

        puts format(
          "Epoch %3d | Train loss %.3f | Val loss %.3f | Val acc %.2f | Time %.3f ms",
          epoch,
          loss.item.to_f,
          val_loss_scalar,
          val_acc.item.to_f,
          (toc - tic) * 1e3
        )
      end

      y_hat = gcn.call(x, adj)
      test_logits = select_rows(y_hat, test_mask)
      test_labels = select_rows(y, test_mask)
      test_loss = loss_fn(test_logits, test_labels)
      test_acc = eval_fn(test_logits, test_labels)
      MLX::Core.eval(test_loss, test_acc)
      puts format("Test loss: %.3f | Test acc: %.2f", test_loss.item.to_f, test_acc.item.to_f)

      gcn
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    hidden_dim: 20,
    dropout: 0.5,
    nb_layers: 2,
    nb_classes: 7,
    bias: true,
    lr: 0.001,
    weight_decay: 0.0,
    patience: 20,
    epochs: 100,
    cpu: false,
    seed: 0,
    synthetic: true,
    synthetic_nodes: 512,
    synthetic_feature_dim: 128,
    synthetic_edge_prob: 0.01,
    data_root: "gcn/data",
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby gcn/main.rb [options]"
    opts.on("--hidden-dim N", Integer, "Hidden layer dimension") { |v| options[:hidden_dim] = v }
    opts.on("--dropout N", Float, "Dropout probability") { |v| options[:dropout] = v }
    opts.on("--nb-layers N", Integer, "Number of hidden GCN layers") { |v| options[:nb_layers] = v }
    opts.on("--nb-classes N", Integer, "Number of classes") { |v| options[:nb_classes] = v }
    opts.on("--bias", "Enable bias in linear layers") { options[:bias] = true }
    opts.on("--no-bias", "Disable bias in linear layers") { options[:bias] = false }
    opts.on("--lr N", Float, "Learning rate") { |v| options[:lr] = v }
    opts.on("--weight-decay N", Float, "L2 regularization factor") { |v| options[:weight_decay] = v }
    opts.on("--patience N", Integer, "Early stopping patience") { |v| options[:patience] = v }
    opts.on("--epochs N", Integer, "Training epochs") { |v| options[:epochs] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
    opts.on("--cpu", "Use CPU device") { options[:cpu] = true }
    opts.on("--synthetic", "Use synthetic graph data") { options[:synthetic] = true }
    opts.on("--real", "Use real Cora dataset") { options[:synthetic] = false }
    opts.on("--synthetic-nodes N", Integer, "Synthetic graph node count") { |v| options[:synthetic_nodes] = v }
    opts.on("--synthetic-feature-dim N", Integer, "Synthetic feature dimension") { |v| options[:synthetic_feature_dim] = v }
    opts.on("--synthetic-edge-prob N", Float, "Synthetic edge probability") { |v| options[:synthetic_edge_prob] = v }
    opts.on("--data-root PATH", String, "Dataset root directory") { |v| options[:data_root] = v }
    opts.on("--python-bin BIN", String, "Python binary for Cora bridge prep") { |v| options[:python_bin] = v }
  end
  parser.parse!

  GcnExample::Train.run(options)
end
