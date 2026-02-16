# frozen_string_literal: true

require "json"
require "optparse"

ROOT = File.expand_path("..", __dir__)

require "mlx"
require "mlx/dsl"
require_relative "hf_bridge"

module BertExample
  class BertConfig
    include MLX::DSL::ConfigSchema

    field :vocab_size, Integer, required: true
    field :hidden_size, Integer, required: true
    field :type_vocab_size, Integer, required: true
    field :max_position_embeddings, Integer, required: true
    field :layer_norm_eps, [Integer, Float], default: 1e-12
    field :num_hidden_layers, Integer, required: true
    field :num_attention_heads, Integer, required: true
    field :intermediate_size, Integer, required: true
  end

  class HfTokenizer
    def initialize(model_name:, bridge:)
      @model_name = model_name
      @bridge = bridge
    end

    def call(batch)
      encoded = @bridge.tokenize(@model_name, batch)
      encoded.each_with_object({}) do |(key, value), out|
        out[key.to_sym] = MLX::Core.array(value, MLX::Core.int32)
      end
    end
  end

  class TransformerEncoderLayer < MLX::DSL::Model
    option :dims
    option :num_heads
    option :mlp_dims, default: -> { dims * 4 }
    option :layer_norm_eps, default: 1e-12

    layer :attention, MLX::NN::MultiHeadAttention, -> { dims }, -> { num_heads }, bias: true
    layer :ln1, MLX::NN::LayerNorm, -> { dims }, eps: -> { layer_norm_eps }
    layer :ln2, MLX::NN::LayerNorm, -> { dims }, eps: -> { layer_norm_eps }
    layer :linear1, MLX::NN::Linear, -> { dims }, -> { mlp_dims }
    layer :linear2, MLX::NN::Linear, -> { mlp_dims }, -> { dims }
    layer :gelu, MLX::NN::GELU

    def call(x, mask = nil, **kwargs)
      mask = kwargs[:mask] if kwargs.key?(:mask)
      attention_out = attention.call(x, x, x, mask)
      add_and_norm = ln1.call(MLX::Core.add(x, attention_out))

      ff = linear1.call(add_and_norm)
      ff = gelu.call(ff)
      ff = linear2.call(ff)
      ln2.call(MLX::Core.add(ff, add_and_norm))
    end
  end

  class TransformerEncoder < MLX::NN::Module
    def initialize(num_layers:, dims:, num_heads:, mlp_dims:, layer_norm_eps:)
      super()
      self.layers = Array.new(num_layers) do
        TransformerEncoderLayer.new(
          dims: dims,
          num_heads: num_heads,
          mlp_dims: mlp_dims,
          layer_norm_eps: layer_norm_eps
        )
      end
    end

    def call(x, mask)
      MLX::DSL.run_stack(layers, x, mask: mask)
    end
  end

  class BertEmbeddings < MLX::DSL::Model
    option :vocab_size
    option :hidden_size
    option :type_vocab_size
    option :max_position_embeddings
    option :layer_norm_eps, default: 1e-12

    layer :word_embeddings, MLX::NN::Embedding, -> { vocab_size }, -> { hidden_size }
    layer :token_type_embeddings, MLX::NN::Embedding, -> { type_vocab_size }, -> { hidden_size }
    layer :position_embeddings, MLX::NN::Embedding, -> { max_position_embeddings }, -> { hidden_size }
    layer :norm, MLX::NN::LayerNorm, -> { hidden_size }, eps: -> { layer_norm_eps }

    def call(input_ids, token_type_ids = nil)
      words = word_embeddings.call(input_ids)
      positions = position_ids_for(input_ids)
      position = position_embeddings.call(positions)
      token_type_ids = MLX::Core.zeros_like(input_ids) if token_type_ids.nil?
      token_types = token_type_embeddings.call(token_type_ids)

      norm.call(MLX::Core.add(MLX::Core.add(position, words), token_types))
    end

    private

    def position_ids_for(input_ids)
      MLX::DSL::Positions.ids_like(input_ids, dtype: MLX::Core.int32)
    end
  end

  class Bert < MLX::DSL::Model
    option :vocab_size
    option :hidden_size
    option :type_vocab_size
    option :max_position_embeddings
    option :layer_norm_eps, default: 1e-12
    option :num_hidden_layers
    option :num_attention_heads
    option :intermediate_size

    layer :embeddings, BertEmbeddings,
          vocab_size: -> { vocab_size },
          hidden_size: -> { hidden_size },
          type_vocab_size: -> { type_vocab_size },
          max_position_embeddings: -> { max_position_embeddings },
          layer_norm_eps: -> { layer_norm_eps }

    layer :encoder, TransformerEncoder,
          num_layers: -> { num_hidden_layers },
          dims: -> { hidden_size },
          num_heads: -> { num_attention_heads },
          mlp_dims: -> { intermediate_size },
          layer_norm_eps: -> { layer_norm_eps }

    layer :pooler, MLX::NN::Linear, -> { hidden_size }, -> { hidden_size }

    def call(input_ids:, token_type_ids: nil, attention_mask: nil)
      x = embeddings.call(input_ids, token_type_ids)

      if !attention_mask.nil?
        attention_mask = attention_mask.astype(MLX::Core.float32)
        attention_mask = MLX::Core.log(attention_mask)
        attention_mask = MLX::Core.expand_dims(attention_mask, 1)
        attention_mask = MLX::Core.expand_dims(attention_mask, 2)
      end

      y = encoder.call(x, attention_mask)
      [y, pooled_output(y)]
    end

    private

    def pooled_output(sequence_output)
      batch_size = sequence_output.shape[0]
      hidden_size = sequence_output.shape[2]
      cls = MLX::Core.slice(sequence_output, [0, 0, 0], [batch_size, 1, hidden_size])
      cls = MLX::Core.squeeze(cls, 1)
      MLX::Core.tanh(pooler.call(cls))
    end
  end

  module_function

  REQUIRED_CONFIG_KEYS = %w[
    vocab_size
    hidden_size
    type_vocab_size
    max_position_embeddings
    layer_norm_eps
    num_hidden_layers
    num_attention_heads
    intermediate_size
  ].freeze

  def load_model(bert_model:, weights_path:, config_path: nil, python_bin: ENV.fetch("PYTHON_BIN", "/usr/bin/env python3"))
    if weights_path.nil? || weights_path.to_s.empty? || !File.exist?(weights_path)
      raise ArgumentError, "No model weights found in #{weights_path.inspect}"
    end

    bridge = PythonBridge.new(python_bin: python_bin)
    config = bridge.config(bert_model, config_path: config_path)
    missing = REQUIRED_CONFIG_KEYS.reject { |key| config.key?(key) }
    unless missing.empty?
      raise ArgumentError, "config is missing required key(s): #{missing.join(', ')}"
    end

    schema = BertConfig.from_hash(config)
    model = Bert.new(
      vocab_size: schema.vocab_size,
      hidden_size: schema.hidden_size,
      type_vocab_size: schema.type_vocab_size,
      max_position_embeddings: schema.max_position_embeddings,
      layer_norm_eps: schema.layer_norm_eps,
      num_hidden_layers: schema.num_hidden_layers,
      num_attention_heads: schema.num_attention_heads,
      intermediate_size: schema.intermediate_size
    )
    model.load_weights(weights_path)

    tokenizer = HfTokenizer.new(model_name: bert_model, bridge: bridge)
    [model, tokenizer]
  end

  def run(bert_model:, mlx_model:, batch:, config_path: nil, python_bin: ENV.fetch("PYTHON_BIN", "/usr/bin/env python3"))
    model, tokenizer = load_model(
      bert_model: bert_model,
      weights_path: mlx_model,
      config_path: config_path,
      python_bin: python_bin
    )
    tokens = tokenizer.call(batch)
    model.call(**tokens)
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    bert_model: "bert-base-uncased",
    mlx_model: "weights/bert-base-uncased.npz",
    text: [],
    config_path: nil,
    python_bin: ENV.fetch("PYTHON_BIN", "/usr/bin/env python3"),
    json_out: nil
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby bert/model.rb [options]"

    opts.on("--bert-model NAME", String, "Hugging Face BERT model name") { |v| options[:bert_model] = v }
    opts.on("--mlx-model PATH", String, "Path to MLX BERT weights (.npz)") { |v| options[:mlx_model] = v }
    opts.on("--config-path PATH", String, "Optional local config JSON path") { |v| options[:config_path] = v }
    opts.on("--text TEXT", String, "Input text (repeatable)") { |v| options[:text] << v }
    opts.on("--python-bin BIN", String, "Python binary for HF bridge") { |v| options[:python_bin] = v }
    opts.on("--json-out PATH", String, "Optional JSON file for full output tensors") { |v| options[:json_out] = v }
  end

  parser.parse!
  options[:text] = options[:text].compact.reject(&:empty?)
  if options[:text].empty?
    options[:text] = ["This is an example of BERT working in MLX Ruby."]
  end

  output, pooled = BertExample.run(
    bert_model: options[:bert_model],
    mlx_model: options[:mlx_model],
    batch: options[:text],
    config_path: options[:config_path],
    python_bin: options[:python_bin]
  )

  puts "output_shape=#{output.shape.inspect}"
  puts "pooled_shape=#{pooled.shape.inspect}"

  if !options[:json_out].nil? && !options[:json_out].empty?
    payload = { "output" => output.to_a, "pooled" => pooled.to_a }
    File.binwrite(options[:json_out], JSON.generate(payload))
    puts "saved_json=#{options[:json_out]}"
  end
end
