# frozen_string_literal: true

require "json"
require "open3"
require "pathname"
require "tmpdir"

module GGUFLLM
  module Utils
    module_function

    SCRIPT_DIR = Pathname.new(__dir__).join("python")
    BUILD_SPM_SCRIPT_PATH = SCRIPT_DIR.join("build_spm_from_metadata.py").to_s
    TOKENIZER_SCRIPT_PATH = SCRIPT_DIR.join("tokenizer_bridge.py").to_s

    class SentencePieceTokenizerBridge
      attr_reader :bos_id, :eos_id

      def initialize(model_path:, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
        @model_path = model_path.to_s
        @python_bin = python_bin
        ids = run_json("ids")
        @bos_id = ids.fetch("bos_id")
        @eos_id = ids.fetch("eos_id")
      end

      def encode(text)
        run_json("encode", text.to_s)
      end

      def decode(tokens)
        run_json("decode", JSON.generate(tokens.map(&:to_i)))
      end

      private

      def run_json(op, arg = "")
        stdout, stderr, status = Open3.capture3(@python_bin, TOKENIZER_SCRIPT_PATH, @model_path, op, arg.to_s)
        return JSON.parse(stdout) if status.success?

        raise RuntimeError, "GGUF tokenizer bridge failed: #{stderr}"
      rescue JSON::ParserError => e
        raise RuntimeError, "GGUF tokenizer bridge returned invalid JSON: #{e.message}"
      end
    end

    def spm_tokenizer(metadata, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      payload = {
        "tokens" => to_ruby_value(metadata.fetch("tokenizer.ggml.tokens")),
        "bos" => scalar(metadata.fetch("tokenizer.ggml.bos_token_id")),
        "eos" => scalar(metadata.fetch("tokenizer.ggml.eos_token_id")),
        "unk" => scalar(metadata.fetch("tokenizer.ggml.unknown_token_id"))
      }
      scores = metadata["tokenizer.ggml.scores"]
      token_types = metadata["tokenizer.ggml.token_type"]
      payload["scores"] = to_ruby_value(scores) unless scores.nil?
      payload["token_types"] = to_ruby_value(token_types) unless token_types.nil?

      stdout, stderr, status = Open3.capture3(
        python_bin,
        BUILD_SPM_SCRIPT_PATH,
        JSON.generate(payload)
      )
      unless status.success?
        raise RuntimeError, "failed to build sentencepiece tokenizer from GGUF metadata: #{stderr}"
      end

      model_path = stdout.strip
      SentencePieceTokenizerBridge.new(model_path: model_path, python_bin: python_bin)
    end

    def scalar(value)
      if value.respond_to?(:item)
        value.item
      else
        value
      end
    end

    def to_ruby_value(value)
      return value.to_a if value.respond_to?(:to_a)

      value
    end
  end
end
