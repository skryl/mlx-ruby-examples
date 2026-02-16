# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

require_relative "models"

module LoraExample
  class SimpleTokenizer
    attr_reader :eos_token_id

    def initialize(vocab_size: 512, add_eos_token: false, eos_token_id: 1)
      @vocab_size = vocab_size
      @add_eos_token = add_eos_token
      @eos_token_id = eos_token_id
    end

    def encode(text)
      ids = text.to_s.bytes.map { |b| 2 + (b % [@vocab_size - 2, 1].max) }
      ids << @eos_token_id if @add_eos_token
      ids
    end

    def decode(tokens)
      tokens.map do |id|
        id_i = id.to_i
        if id_i < 2
          ""
        else
          (((id_i - 2) % 95) + 32).chr
        end
      end.join
    end

    def save(path)
      payload = {
        "vocab_size" => @vocab_size,
        "add_eos_token" => @add_eos_token,
        "eos_token_id" => @eos_token_id
      }
      File.binwrite(path, JSON.pretty_generate(payload))
    end

    def self.load(path)
      payload = JSON.parse(File.binread(path))
      new(
        vocab_size: payload.fetch("vocab_size"),
        add_eos_token: payload.fetch("add_eos_token", false),
        eos_token_id: payload.fetch("eos_token_id", 1)
      )
    end
  end

  class HfTokenizer
    SCRIPT_PATH = Pathname.new(__dir__).join("python", "tokenizer_bridge.py").to_s

    attr_reader :eos_token_id

    def initialize(model_name, add_eos_token: false, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @model_name = model_name.to_s
      @python_bin = python_bin
      @add_eos_token = add_eos_token
      @eos_token_id = run_json("eos")
    end

    def encode(text)
      run_json("encode", text.to_s)
    end

    def decode(tokens)
      run_json("decode", JSON.generate(tokens.map(&:to_i)))
    end

    private

    def run_json(op, arg = "")
      stdout, stderr, status = Open3.capture3(
        @python_bin,
        SCRIPT_PATH,
        @model_name,
        op.to_s,
        @add_eos_token ? "1" : "0",
        arg.to_s
      )
      return JSON.parse(stdout) if status.success?

      raise RuntimeError, "tokenizer bridge failed: #{stderr}"
    rescue JSON::ParserError => e
      raise RuntimeError, "tokenizer bridge returned invalid JSON: #{e.message}"
    end
  end

  module Utils
    module_function

    def default_synthetic_model_args(vocab_size: 512)
      ModelArgs.new(
        hidden_size: 64,
        num_hidden_layers: 2,
        intermediate_size: 128,
        num_attention_heads: 4,
        rms_norm_eps: 1e-5,
        vocab_size: vocab_size,
        num_key_value_heads: 4,
        rope_theta: 10_000.0,
        rope_traditional: false
      )
    end

    def load(path_or_hf_repo, tokenizer_config = {}, synthetic: false, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      if synthetic
        args = default_synthetic_model_args
        model = Model.new(args)
        tokenizer = SimpleTokenizer.new(
          vocab_size: args.vocab_size,
          add_eos_token: tokenizer_config.fetch("add_eos_token", false)
        )
        return [model, tokenizer, args.to_h]
      end

      model_path = Pathname.new(path_or_hf_repo.to_s).expand_path
      unless model_path.exist?
        raise Errno::ENOENT, "Model path does not exist: #{model_path}. Convert first before fine-tuning."
      end

      config_path = model_path.join("config.json")
      raise Errno::ENOENT, "Missing config.json in #{model_path}" unless config_path.exist?

      config = JSON.parse(File.binread(config_path))
      model_args = ModelArgs.from_dict(config)
      model = Model.new(model_args)

      weights = load_weight_files(model_path)
      model.update(MLX::Utils.tree_unflatten(weights.to_a))
      MLX::Core.eval(model.parameters)

      tokenizer = if model_path.join("tokenizer_simple.json").exist?
        SimpleTokenizer.load(model_path.join("tokenizer_simple.json"))
      else
        HfTokenizer.new(
          model_path.to_s,
          add_eos_token: tokenizer_config.fetch("add_eos_token", false),
          python_bin: python_bin
        )
      end
      [model, tokenizer, config]
    end

    def load_weight_files(model_path)
      files = []
      unsharded = model_path.join("weights.npz")
      files << unsharded.to_s if unsharded.exist?
      files.concat(Dir.glob(model_path.join("weights.*.npz").to_s).sort)
      files.concat(Dir.glob(model_path.join("*.safetensors").to_s).sort)
      raise Errno::ENOENT, "No weight files found in #{model_path}" if files.empty?

      files.each_with_object({}) do |file, out|
        MLX::Core.load(file).to_a.each do |key, value|
          out[key.to_s] = value
        end
      end
    end

    def save_model(save_dir, weights, tokenizer, config)
      save_path = Pathname.new(save_dir.to_s).expand_path
      save_path.mkpath

      payload = {}
      weights.each do |key, value|
        payload[key.to_s.to_sym] = value
      end
      MLX::Core.savez(save_path.join("weights.npz").to_s, **payload)
      File.binwrite(save_path.join("config.json"), JSON.pretty_generate(config))

      if tokenizer.is_a?(SimpleTokenizer)
        tokenizer.save(save_path.join("tokenizer_simple.json"))
      end
    end

    def generate(prompt, model, temp: 0.0)
      sample = lambda do |logits|
        if temp.to_f.zero?
          MLX::Core.argmax(logits, -1)
        else
          MLX::Core.categorical(MLX::Core.multiply(logits, 1.0 / temp.to_f))
        end
      end

      y = prompt
      cache = nil
      Enumerator.new do |enum|
        loop do
          logits, cache = model.call(MLX::Core.expand_dims(y, 0), cache: cache)
          idx = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
          logits = MLX::Core.take(logits, idx, 1)
          logits = MLX::Core.squeeze(logits, 1)
          y = sample.call(logits).astype(MLX::Core.int32)
          enum << MLX::Core.squeeze(y)
        end
      end
    end
  end
end
