# frozen_string_literal: true

require "json"
require "open3"
require "optparse"
require "pathname"

require_relative "llava"

module LlavaExample
  class LlavaDecodeAdapter
    def initialize(model:, pixel_values:)
      @model = model
      @pixel_values = pixel_values
    end

    def call(input_ids, cache: nil, **kwargs)
      if cache.nil?
        @model.call(input_ids, @pixel_values, cache: nil)
      else
        @model.language_model.call(input_ids, cache: cache, **kwargs)
      end
    end
  end

  class ProcessorBridge
    SCRIPT_PATH = Pathname.new(__dir__).join("python", "processor_bridge.py").to_s

    attr_reader :eos_token_id

    def initialize(model_path:, tokenizer_config: {}, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @python_bin = python_bin
      @model_path = model_path.to_s
      @tokenizer_config = tokenizer_config

      meta = run_json("meta", "tokenizer_config" => @tokenizer_config)
      @eos_token_id = meta["eos_token_id"]
    end

    def prepare_inputs(image:, prompt:)
      payload = {
        "tokenizer_config" => @tokenizer_config,
        "image" => image.to_s,
        "prompt" => prompt.to_s
      }
      out = run_json("prepare", payload)
      pixel_values = MLX::Core.array(out.fetch("pixel_values"), MLX::Core.float32)
      input_ids = MLX::Core.array(out.fetch("input_ids"), MLX::Core.int32)
      [pixel_values, input_ids]
    end

    def decode(tokens)
      payload = {
        "tokenizer_config" => @tokenizer_config,
        "tokens" => tokens.map(&:to_i)
      }
      run_json("decode", payload)
    end

    private

    def run_json(op, payload = {})
      stdout, stderr, status = Open3.capture3(
        @python_bin,
        SCRIPT_PATH,
        @model_path,
        op.to_s,
        JSON.generate(payload)
      )
      return JSON.parse(stdout) if status.success?

      raise RuntimeError, "processor bridge failed: #{stderr}"
    rescue JSON::ParserError => e
      raise RuntimeError, "processor bridge returned invalid JSON: #{e.message}"
    end
  end

  module_function

  def sample(logits, temperature: 0.0)
    if temperature.to_f.zero?
      MLX::Core.argmax(logits, -1)
    else
      MLX::Core.categorical(MLX::Core.multiply(logits, 1.0 / temperature.to_f))
    end
  end

  def last_logits(logits)
    last_idx = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
    MLX::Core.squeeze(MLX::Core.take(logits, last_idx, 1), 1)
  end

  def generate_text(input_ids:, pixel_values:, model:, processor:, max_tokens:, temperature:)
    sampler = if temperature.to_f.zero?
      { strategy: :argmax }
    else
      { strategy: :temperature, temperature: temperature.to_f }
    end
    adapter = LlavaDecodeAdapter.new(model: model, pixel_values: pixel_values)
    generator = MLX::DSL::Generate.new(
      model: adapter,
      eos_id: processor.eos_token_id,
      sampler: sampler,
      mode: :decoder_only
    )

    tokens = []
    generator.each_token(input_ids: input_ids, max_tokens: max_tokens) do |token_id, _chunk|
      tokens << token_id.to_i
    end

    processor.decode(tokens)
  end

  def load_model(model_path, tokenizer_config: {}, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    processor = ProcessorBridge.new(
      model_path: model_path,
      tokenizer_config: tokenizer_config,
      python_bin: python_bin
    )
    model = LlavaModel.from_pretrained(model_path, python_bin: python_bin)
    [processor, model]
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    model: "llava-hf/llava-1.5-7b-hf",
    image: "http://images.cocodataset.org/val2017/000000039769.jpg",
    prompt: "USER: <image>\\nWhat are these?\\nASSISTANT:",
    max_tokens: 100,
    temp: 0.3,
    eos_token: nil,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llava/generate.rb [options]"
    opts.on("--model PATH", String, "Local model directory or Hugging Face repo") { |v| options[:model] = v }
    opts.on("--image PATH_OR_URL", String, "Image path or URL") { |v| options[:image] = v }
    opts.on("--prompt TEXT", String, "Prompt (supports escaped \\n)") { |v| options[:prompt] = v }
    opts.on("--max-tokens N", Integer, "Max generated tokens") { |v| options[:max_tokens] = v }
    opts.on("--temp N", Float, "Sampling temperature (0.0 = greedy)") { |v| options[:temp] = v }
    opts.on("--eos-token TEXT", String, "Optional tokenizer EOS token override") { |v| options[:eos_token] = v }
    opts.on("--python-bin BIN", String, "Python binary for processor/snapshot bridges") { |v| options[:python_bin] = v }
  end
  parser.parse!

  tokenizer_config = {}
  tokenizer_config["eos_token"] = options[:eos_token] unless options[:eos_token].nil?

  processor, model = LlavaExample.load_model(
    options[:model],
    tokenizer_config: tokenizer_config,
    python_bin: options[:python_bin]
  )

  prompt = options[:prompt].gsub("\\n", "\n")
  pixel_values, input_ids = processor.prepare_inputs(image: options[:image], prompt: prompt)

  puts prompt
  generated = LlavaExample.generate_text(
    input_ids: input_ids,
    pixel_values: pixel_values,
    model: model,
    processor: processor,
    max_tokens: options[:max_tokens],
    temperature: options[:temp]
  )
  puts generated
end
