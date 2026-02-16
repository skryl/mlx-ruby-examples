# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

ROOT = File.expand_path("../..", __dir__)

require "mlx"
require "mlx/dsl"

require_relative "utils"

module GGUFLLM
  SCRIPT_DIR = Pathname.new(__dir__).join("python")

  class ModelArgs
    include MLX::DSL::ConfigSchema

    field :hidden_size, Integer, required: true
    field :num_hidden_layers, Integer, required: true
    field :intermediate_size, Integer, required: true
    field :num_attention_heads, Integer, required: true
    field :rms_norm_eps, [Integer, Float], required: true
    field :vocab_size, Integer, required: true
    field :context_length, Integer, required: true
    field :num_key_value_heads, Integer, default: ->(cfg) { cfg.num_attention_heads }
    field :rope_theta, [Integer, Float], default: 10_000.0
    field :rope_traditional, [TrueClass, FalseClass], default: false
    field :model_type, [String, NilClass], default: nil
    field :rope_scaling, [Hash, NilClass], default: nil do |value|
      next if value.nil?
      required = %w[factor type]
      unless required.all? { |key| value.key?(key) || value.key?(key.to_sym) }
        raise ArgumentError, "rope_scaling must contain keys #{required.inspect}"
      end
      kind = value["type"] || value[:type]
      raise ArgumentError, "rope_scaling 'type' currently only supports 'linear'" unless kind == "linear"
    end
  end

  class Attention < MLX::NN::Module
    def initialize(args)
      super()
      dim = args.hidden_size
      @n_heads = args.num_attention_heads
      @n_kv_heads = args.num_key_value_heads || @n_heads
      @head_dim = args.hidden_size / @n_heads
      @scale = @head_dim**-0.5

      self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
      self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
      self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
      self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

      rope_scale = if !args.rope_scaling.nil? && (args.rope_scaling["type"] || args.rope_scaling[:type]) == "linear"
        1.0 / (args.rope_scaling["factor"] || args.rope_scaling[:factor]).to_f
      else
        1.0
      end
      self.rope = MLX::NN::RoPE.new(
        @head_dim,
        traditional: args.rope_traditional,
        base: args.rope_theta,
        scale: rope_scale
      )
    end

    def call(x, mask: nil, cache: nil)
      batch_size, seq_len, _dims = x.shape
      queries = q_proj.call(x)
      keys = k_proj.call(x)
      values = v_proj.call(x)

      queries = MLX::Core.transpose(
        MLX::Core.reshape(queries, [batch_size, seq_len, @n_heads, @head_dim]),
        [0, 2, 1, 3]
      )
      keys = MLX::Core.transpose(
        MLX::Core.reshape(keys, [batch_size, seq_len, @n_kv_heads, @head_dim]),
        [0, 2, 1, 3]
      )
      values = MLX::Core.transpose(
        MLX::Core.reshape(values, [batch_size, seq_len, @n_kv_heads, @head_dim]),
        [0, 2, 1, 3]
      )

      if !cache.nil?
        key_cache, value_cache = cache
        offset = key_cache.shape[2]
        queries = rope.call(queries, offset: offset)
        keys = rope.call(keys, offset: offset)
        keys = MLX::Core.concatenate([key_cache, keys], 2)
        values = MLX::Core.concatenate([value_cache, values], 2)
      else
        queries = rope.call(queries)
        keys = rope.call(keys)
      end

      output = MLX::Core.scaled_dot_product_attention(queries, keys, values, @scale, mask)
      output = MLX::Core.transpose(output, [0, 2, 1, 3])
      output = MLX::Core.reshape(output, [batch_size, seq_len, @n_heads * @head_dim])
      [o_proj.call(output), [keys, values]]
    end
  end

  class MLP < MLX::NN::Module
    def initialize(dim, hidden_dim)
      super()
      self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
      self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
      self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
    end

    def call(x)
      down_proj.call(MLX::Core.multiply(MLX::NN.silu(gate_proj.call(x)), up_proj.call(x)))
    end
  end

  class TransformerBlock < MLX::NN::Module
    def initialize(args)
      super()
      self.self_attn = Attention.new(args)
      self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
      self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
      self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
    end

    def call(x, mask: nil, cache: nil)
      residual, next_cache = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
      hidden = MLX::Core.add(x, residual)
      residual = mlp.call(post_attention_layernorm.call(hidden))
      [MLX::Core.add(hidden, residual), next_cache]
    end
  end

  class LlamaModel < MLX::NN::Module
    attr_reader :vocab_size, :num_hidden_layers

    def initialize(args)
      super()
      @vocab_size = args.vocab_size
      @num_hidden_layers = args.num_hidden_layers
      raise ArgumentError, "vocab_size must be positive" if @vocab_size <= 0

      self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
      self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
      self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
      puts <<~INFO
        Model info
        ==========
        Context length: #{args.context_length}
        Vocab size: #{args.vocab_size}
        Hidden size: #{args.hidden_size}
        Num layers: #{args.num_hidden_layers}
        Num attention heads: #{args.num_attention_heads}
      INFO
    end

    def call(inputs, cache: nil)
      hidden = embed_tokens.call(inputs)
      offset = MLX::DSL::Positions.offset_from_cache(cache, layer: 0)

      mask = nil
      if hidden.shape[1] > 1 || offset.positive?
        mask = MLX::DSL::Masks.causal(length: hidden.shape[1], offset: offset, dtype: hidden.dtype)
      end

      cache_state = cache || Array.new(layers.length)
      hidden, next_cache = MLX::DSL.run_stack(layers, hidden, mask: mask, cache: cache_state)

      [norm.call(hidden), next_cache]
    end
  end

  class Model < MLX::NN::Module
    def initialize(args)
      super()
      self.model = LlamaModel.new(args)
      self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
    end

    def call(inputs, cache: nil)
      out, cache = model.call(inputs, cache: cache)
      [lm_head.call(out), cache]
    end
  end

  class GGUFTokenizer
    def initialize(metadata, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @tokenizer = Utils.spm_tokenizer(metadata, python_bin: python_bin)
    end

    def encode(text)
      MLX::Core.array([@tokenizer.bos_id] + @tokenizer.encode(text.to_s), MLX::Core.int32)
    end

    def eos_token_id
      @tokenizer.eos_id
    end

    def decode(tokens)
      @tokenizer.decode(tokens)
    end
  end

  module_function

  def scalar(value)
    if value.respond_to?(:item)
      value.item
    else
      value
    end
  end

  def get_config(metadata)
    tokens = metadata.fetch("tokenizer.ggml.tokens")
    token_list = tokens.respond_to?(:to_a) ? tokens.to_a : tokens
    {
      "context_length" => scalar(metadata.fetch("llama.context_length")),
      "hidden_size" => scalar(metadata.fetch("llama.embedding_length")),
      "num_hidden_layers" => scalar(metadata.fetch("llama.block_count")),
      "num_attention_heads" => scalar(metadata.fetch("llama.attention.head_count")),
      "intermediate_size" => scalar(metadata.fetch("llama.feed_forward_length")),
      "num_key_value_heads" => scalar(metadata.fetch("llama.attention.head_count_kv")),
      "rms_norm_eps" => scalar(metadata.fetch("llama.attention.layer_norm_rms_epsilon")),
      "vocab_size" => token_list.length,
      "rope_theta" => scalar(metadata.fetch("llama.rope.freq_base")),
      "rope_traditional" => true
    }
  end

  def weight_mapper
    MLX::DSL.weight_map do
      rename "blk." => "model.layers."
      rename "ffn_gate" => "mlp.gate_proj"
      rename "ffn_down" => "mlp.down_proj"
      rename "ffn_up" => "mlp.up_proj"
      rename "attn_q" => "self_attn.q_proj"
      rename "attn_k" => "self_attn.k_proj"
      rename "attn_v" => "self_attn.v_proj"
      rename "attn_output" => "self_attn.o_proj"
      rename "attn_norm" => "input_layernorm"
      rename "ffn_norm" => "post_attention_layernorm"
      rename "token_embd" => "model.embed_tokens"
      rename "output_norm" => "model.norm"
      rename "output" => "lm_head"
    end
  end

  # Backward-compatible key translation helper used by existing tests/tools.
  def translate_weight_names(name)
    mapped = weight_mapper.apply({name.to_s => nil})
    mapped.keys.first
  end

  def snapshot_download(repo:, gguf_file:, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    script_path = SCRIPT_DIR.join("snapshot_download.py").to_s
    stdout, stderr, status = Open3.capture3(python_bin, script_path, repo, gguf_file)
    unless status.success?
      raise RuntimeError, "Failed to download #{gguf_file} from #{repo}: #{stderr}"
    end
    stdout.strip
  end

  def load(gguf_file, repo = nil, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    gguf_path = Pathname.new(gguf_file.to_s)
    unless gguf_path.exist?
      if repo.nil?
        raise ArgumentError, "Could not find file #{gguf_file}, and no Hugging Face repo provided for download."
      end
      model_path = snapshot_download(repo: repo, gguf_file: gguf_path.to_s, python_bin: python_bin)
      candidate = Pathname.new(model_path).join(gguf_path.to_s)
      unless candidate.exist?
        raise ArgumentError, "File #{gguf_file} not in repo #{repo}."
      end
      gguf_path = candidate
    end

    puts "[INFO] Loading model from #{gguf_path}"
    weights, metadata = MLX::Core.load(gguf_path.to_s, nil, true)
    gguf_ft = scalar(metadata.fetch("general.file_type"))
    quantization = case gguf_ft
    when 0, 1
      nil
    when 2, 3
      puts "4 bits quantized model"
      { group_size: 32, bits: 4 }
    when 7
      puts "8 bits quantized model"
      { group_size: 32, bits: 8 }
    else
      puts "[WARNING] Using unsupported GGUF quantization. Casting to float16."
      nil
    end

    renamed = weight_mapper.apply(weights)

    config = get_config(metadata)
    model = Model.new(ModelArgs.from_hash(config))

    if !quantization.nil?
      class_predicate = lambda do |path, module_obj|
        (module_obj.is_a?(MLX::NN::Linear) || module_obj.is_a?(MLX::NN::Embedding)) && renamed.key?("#{path}.scales")
      end
      MLX::NN.quantize(model, **quantization, class_predicate: class_predicate)
    end

    tokenizer = GGUFTokenizer.new(metadata, python_bin: python_bin)
    model.load_weights(renamed.to_a, strict: false)
    [model, tokenizer]
  end

  def generate(prompt, model, temp: 0.0, max_tokens: 1_000_000)
    sampler = if temp.to_f.zero?
      { strategy: :argmax }
    else
      { strategy: :temperature, temperature: temp.to_f }
    end
    generator = MLX::DSL::Generate.new(model: model, sampler: sampler, mode: :decoder_only)
    Enumerator.new do |emitter|
      generator.each_token(input_ids: prompt, max_tokens: max_tokens) do |token_id, _chunk|
        emitter << MLX::Core.array(token_id, MLX::Core.int32)
      end
    end
  end
end
