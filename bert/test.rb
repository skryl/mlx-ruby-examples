# frozen_string_literal: true

require "json"
require "optparse"

require_relative "model"

module BertExample
  module Parity
    module_function

    def allclose?(left, right, rtol:, atol:)
      lhs = flatten_numeric(left)
      rhs = flatten_numeric(right)
      return false unless lhs.length == rhs.length

      lhs.each_with_index do |a, i|
        b = rhs[i]
        tolerance = atol + (rtol * b.abs)
        return false if (a - b).abs > tolerance
      end
      true
    end

    def flatten_numeric(value, out = [])
      if value.is_a?(Array)
        value.each { |item| flatten_numeric(item, out) }
      else
        out << value.to_f
      end
      out
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    integration: false,
    seed: 23,
    bert_model: "bert-base-uncased",
    mlx_model: "weights/bert-base-uncased.npz",
    config_path: nil,
    text: [],
    python_bin: ENV.fetch("PYTHON_BIN", "/usr/bin/env python3"),
    rtol: 1e-4,
    atol: 1e-5
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby bert/test.rb [options]"
    opts.on("--integration", "Run Hugging Face parity integration test") { options[:integration] = true }
    opts.on("--seed N", Integer, "PRNG seed for synthetic test path") { |v| options[:seed] = v }
    opts.on("--bert-model NAME", String, "Hugging Face model name") { |v| options[:bert_model] = v }
    opts.on("--mlx-model PATH", String, "Path to converted NPZ weights") { |v| options[:mlx_model] = v }
    opts.on("--config-path PATH", String, "Optional local config path") { |v| options[:config_path] = v }
    opts.on("--text TEXT", String, "Input text (repeatable)") { |v| options[:text] << v }
    opts.on("--python-bin BIN", String, "Python binary for HF bridge") { |v| options[:python_bin] = v }
    opts.on("--rtol N", Float, "Relative tolerance") { |v| options[:rtol] = v }
    opts.on("--atol N", Float, "Absolute tolerance") { |v| options[:atol] = v }
  end
  parser.parse!

  if options[:integration]
    options[:text] = options[:text].compact.reject(&:empty?)
    if options[:text].empty?
      options[:text] = ["This is an example of BERT working in MLX Ruby."]
    end

    bridge = BertExample::PythonBridge.new(python_bin: options[:python_bin])
    torch_out = bridge.torch_forward(options[:bert_model], options[:text])

    mlx_output, mlx_pooled = BertExample.run(
      bert_model: options[:bert_model],
      mlx_model: options[:mlx_model],
      batch: options[:text],
      config_path: options[:config_path],
      python_bin: options[:python_bin]
    )

    output_ok = BertExample::Parity.allclose?(
      torch_out.fetch("last_hidden_state"),
      mlx_output.to_a,
      rtol: options[:rtol],
      atol: options[:atol]
    )
    unless output_ok
      raise "BERT sequence output mismatch against torch reference"
    end

    pooled_reference = torch_out["pooler_output"]
    if !pooled_reference.nil?
      pooled_ok = BertExample::Parity.allclose?(
        pooled_reference,
        mlx_pooled.to_a,
        rtol: options[:rtol],
        atol: options[:atol]
      )
      unless pooled_ok
        raise "BERT pooled output mismatch against torch reference"
      end
    end
  else
    MLX::Core.random_seed(options[:seed])
    model = BertExample::Bert.new(
      vocab_size: 64,
      hidden_size: 32,
      type_vocab_size: 2,
      max_position_embeddings: 32,
      layer_norm_eps: 1e-12,
      num_hidden_layers: 2,
      num_attention_heads: 4,
      intermediate_size: 64
    )

    input_ids = MLX::Core.random_uniform([2, 8], 0.0, 63.0, MLX::Core.float32).astype(MLX::Core.int32)
    token_type_ids = MLX::Core.zeros_like(input_ids)
    attention_mask = MLX::Core.ones_like(input_ids)
    output, pooled = model.call(
      input_ids: input_ids,
      token_type_ids: token_type_ids,
      attention_mask: attention_mask
    )
    MLX::Core.eval(output, pooled)

    unless output.shape == [2, 8, 32]
      raise "BERT synthetic output shape mismatch: #{output.shape.inspect}"
    end
    unless pooled.shape == [2, 32]
      raise "BERT synthetic pooled shape mismatch: #{pooled.shape.inspect}"
    end
  end

  puts "Tests pass :)"
end
