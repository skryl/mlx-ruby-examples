# frozen_string_literal: true

require "optparse"
require "securerandom"
require "tmpdir"
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
      best_ckpt = options[:best_ckpt]
      created_temp_checkpoint = false
      if best_ckpt.nil? || best_ckpt.empty?
        best_ckpt = File.join(Dir.tmpdir, "gcn_best_#{SecureRandom.hex(8)}.npz")
        created_temp_checkpoint = true
      end

      trainer = gcn.trainer(optimizer: optimizer) do |features:, graph_adj:, labels:, mask:|
        forward_loss(gcn, features, graph_adj, labels, mask, options[:weight_decay])
      end
      artifact_kwargs = {
        checkpoint: {
          path: best_ckpt,
          strategy: :best
        },
        retention: { keep_last_n: 1 }
      }
      resume_source = options[:resume_from]
      artifact_kwargs[:resume] = resume_source unless resume_source.nil? || resume_source.empty?
      run_bundle_path = options[:run_bundle_path]
      unless run_bundle_path.nil? || run_bundle_path.empty?
        artifact_kwargs[:run_bundle] = {
          enabled: true,
          path: run_bundle_path,
          config: {
            "example" => "gcn",
            "hidden_dim" => options[:hidden_dim],
            "nb_layers" => options[:nb_layers]
          }
        }
      end
      trainer.artifact_policy(**artifact_kwargs)

      epoch_started_at = {}
      trainer.before_epoch do |ctx|
        epoch_started_at[ctx.fetch(:epoch)] = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      end
      trainer.after_epoch do |ctx|
        epoch = ctx.fetch(:epoch)
        started_at = epoch_started_at.fetch(epoch, Process.clock_gettime(Process::CLOCK_MONOTONIC))
        y_hat = gcn.call(x, adj)
        val_acc = eval_fn(select_rows(y_hat, val_mask), select_rows(y, val_mask))
        MLX::Core.eval(val_acc)
        puts format(
          "Epoch %3d | Train loss %.3f | Val loss %.3f | Val acc %.2f | Time %.3f ms",
          epoch,
          ctx.fetch(:epoch_loss).to_f,
          ctx.fetch(:val_loss).to_f,
          val_acc.item.to_f,
          (Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at) * 1e3
        )
      end

      train_source = lambda do |epoch:, **_kwargs|
        _ = epoch
        [
          {
            features: x,
            graph_adj: adj,
            labels: y,
            mask: train_mask
          }
        ]
      end
      val_source = lambda do |epoch:, **_kwargs|
        _ = epoch
        [
          {
            features: x,
            graph_adj: adj,
            labels: y,
            mask: val_mask
          }
        ]
      end
      trainer.register_dataflow(
        :graph_cls,
        train: { reduce: :mean },
        validation: { reduce: :mean }
      )
      split_plan = MLX::DSL.splits do
        train(train_source)
        validation(val_source)
      end

      trainer.fit_report(
        split_plan,
        **trainer.use_dataflow(:graph_cls),
        epochs: options[:epochs],
        monitor: :val_loss,
        monitor_mode: :min,
        patience: options[:patience],
        min_delta: options.fetch(:min_delta, 0.0),
        keep_losses: false
      )

      gcn.load_checkpoint(best_ckpt, optimizer: optimizer) if File.exist?(best_ckpt)

      y_hat = gcn.call(x, adj)
      test_logits = select_rows(y_hat, test_mask)
      test_labels = select_rows(y, test_mask)
      test_loss = loss_fn(test_logits, test_labels)
      test_acc = eval_fn(test_logits, test_labels)
      MLX::Core.eval(test_loss, test_acc)
      puts format("Test loss: %.3f | Test acc: %.2f", test_loss.item.to_f, test_acc.item.to_f)

      gcn
    ensure
      if created_temp_checkpoint && !best_ckpt.nil? && File.exist?(best_ckpt)
        File.delete(best_ckpt)
      end
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
    min_delta: 0.0,
    epochs: 100,
    cpu: false,
    seed: 0,
    synthetic: true,
    synthetic_nodes: 512,
    synthetic_feature_dim: 128,
    synthetic_edge_prob: 0.01,
    data_root: "gcn/data",
    python_bin: ENV.fetch("PYTHON_BIN", "python3"),
    best_ckpt: nil,
    run_bundle_path: nil,
    resume_from: nil
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
    opts.on("--min-delta N", Float, "Minimum val-loss delta for improvement") { |v| options[:min_delta] = v }
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
    opts.on("--best-ckpt PATH", String, "Path for best checkpoint artifact") { |v| options[:best_ckpt] = v }
    opts.on("--run-bundle PATH", String, "Auto-save DSL run bundle path") { |v| options[:run_bundle_path] = v }
    opts.on("--resume-from SOURCE", String, "Resume source (checkpoint or run bundle path)") { |v| options[:resume_from] = v }
  end
  parser.parse!

  GcnExample::Train.run(options)
end
