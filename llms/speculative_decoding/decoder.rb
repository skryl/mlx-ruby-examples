# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

require_relative "model"

module SpeculativeDecodingExample
  class Tokenizer
    attr_reader :eos_id, :decoder_start_id
    SCRIPT_PATH = Pathname.new(__dir__).join("python", "tokenizer_bridge.py").to_s

    def initialize(model_name, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @model_name = model_name.to_s
      @python_bin = python_bin
      @eos_id = run_json("eos")
      @decoder_start_id = 0
    end

    def encode(text)
      run_json("encode", text.to_s)
    end

    def decode(tokens)
      run_json("decode", JSON.generate(tokens))
    end

    private

    def run_json(op, arg = "")
      stdout, stderr, status = Open3.capture3(@python_bin, SCRIPT_PATH, @model_name, op, arg.to_s)
      return JSON.parse(stdout) if status.success?

      raise RuntimeError, "tokenizer bridge failed: #{stderr}"
    rescue JSON::ParserError => e
      raise RuntimeError, "tokenizer bridge returned invalid JSON: #{e.message}"
    end
  end

  class SpeculativeDecoder
    attr_reader :tokenizer, :model, :draft_model, :num_draft, :delta

    def initialize(model:, draft_model:, tokenizer:, num_draft: 5, delta: 0.0, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @tokenizer = Tokenizer.new(tokenizer, python_bin: python_bin)
      @model = model
      @draft_model = draft_model
      @num_draft = num_draft
      @delta = delta
    end

    def generate(prompt, max_tokens: 100)
      memory = model.encode(MLX::Core.expand_dims(MLX::Core.array(tokenizer.encode(prompt), MLX::Core.int32), 0))
      x = MLX::Core.array([tokenizer.decoder_start_id], MLX::Core.int32)
      skip = 0
      outputs = []

      generate_stream(x, memory: memory).each_with_index do |(token, _), n|
        break if token.item.to_i == tokenizer.eos_id

        outputs << token.item.to_i
        if ((n + 1) % 10).zero?
          str_output = tokenizer.decode(outputs)
          chunk = str_output[skip..]
          print(chunk, end: "", flush: true) unless chunk.nil?
          skip = str_output.length
        end
        break if n + 1 >= max_tokens
      end

      tail = tokenizer.decode(outputs)
      chunk = tail[skip..]
      print(chunk, end: "", flush: true) unless chunk.nil?
      puts
      model.reset_cache
    end

    def speculative_decode(prompt, max_tokens: 100)
      sample = lambda do |logits|
        MLX::Core.argmax(logits, -1)
      end

      prompt_tokens = tokenizer.encode(prompt)
      prompt_array = MLX::Core.expand_dims(MLX::Core.array(prompt_tokens, MLX::Core.int32), 0)
      memory = model.encode(prompt_array)
      draft_memory = draft_model.encode(prompt_array)
      tokens = MLX::Core.array([tokenizer.decoder_start_id], MLX::Core.int32)

      n_steps = 0
      ntoks = 0
      n_accepted = 0
      n_draft_total = 0
      outputs = []
      skip = 0
      draft_inputs = tokens
      inputs = tokens

      loop do
        draft_tokens = []
        draft_probs = []
        draft_budget = [num_draft, max_tokens - ntoks].min
        break if draft_budget <= 0

        draft_gen = generate_stream(draft_inputs, memory: draft_memory, draft: true)
        draft_budget.times do
          t, p = draft_gen.next
          draft_tokens << t
          draft_probs << p
          break if t.item.to_i == tokenizer.eos_id
        end

        draft_tokens = MLX::Core.concatenate(draft_tokens)
        draft_probs = MLX::Core.concatenate(draft_probs)
        verify_tokens = MLX::Core.concatenate([inputs, draft_tokens])
        logits = model.decode(
          MLX::Core.expand_dims(verify_tokens, 0),
          memory
        )
        logits = MLX::Core.squeeze(logits, 0)

        num_to_accept = get_num_accept(
          draft_tokens,
          draft_probs,
          prefix_rows(logits, logits.shape[0] - 1)
        )
        new_tokens = prefix_1d(draft_tokens, num_to_accept)
        model_token = sample.call(logits[num_to_accept])
        model_token = MLX::Core.expand_dims(model_token, 0)
        new_tokens = if num_to_accept.zero?
          model_token
        else
          MLX::Core.concatenate([new_tokens, model_token])
        end

        n_accepted += num_to_accept
        n_draft_total += draft_tokens.shape[0]

        if draft_tokens.shape[0] > num_to_accept
          draft_model.truncate_cache(draft_tokens.shape[0] - new_tokens.shape[0])
          model.truncate_cache(draft_tokens.shape[0] - new_tokens.shape[0] + 1)
        end

        n_steps += 1

        new_token_list = new_tokens.to_a.map(&:to_i)
        new_token_list.each do |token_id|
          break if token_id == tokenizer.eos_id || ntoks >= max_tokens

          outputs << token_id
          ntoks += 1
        end

        str_output = tokenizer.decode(outputs)
        chunk = str_output[skip..]
        print(chunk, end: "", flush: true) unless chunk.nil?
        skip = str_output.length

        break if ntoks >= max_tokens || new_token_list[-1] == tokenizer.eos_id

        draft_inputs = tail_1d(new_tokens, [2, new_tokens.shape[0]].min)
        inputs = tail_1d(draft_inputs, 1)
      end

      tail = tokenizer.decode(outputs)
      chunk = tail[skip..]
      print(chunk, end: "", flush: true) unless chunk.nil?
      puts

      model.reset_cache
      draft_model.reset_cache
      { "n_accepted" => n_accepted, "n_draft" => n_draft_total, "n_steps" => n_steps }
    end

    private

    def generate_stream(x, memory:, draft: false)
      chosen_model = draft ? draft_model : model
      Enumerator.new do |emitter|
        loop do
          logits = chosen_model.decode(MLX::Core.expand_dims(x, 0), memory)
          logits = last_logits(logits)
          x = MLX::Core.argmax(logits, -1, true).astype(MLX::Core.int32)
          lognorm = MLX::Core.logsumexp(logits.astype(MLX::Core.float32))
          logprob = MLX::Core.subtract(MLX::Core.take(logits, x, 0), lognorm)
          emitter << [x, logprob]
        end
      end
    end

    def last_logits(logits)
      last_idx = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
      last = MLX::Core.take(logits, last_idx, 1)
      last = MLX::Core.squeeze(last, 1)
      MLX::Core.squeeze(last, 0)
    end

    def get_num_accept(draft_tokens, draft_probs, model_logits)
      model_probs = MLX::Core.take_along_axis(
        model_logits,
        MLX::Core.expand_dims(draft_tokens, 1),
        -1
      )
      model_probs = MLX::Core.squeeze(model_probs, 1)
      model_probs = MLX::Core.subtract(
        model_probs,
        MLX::Core.logsumexp(model_logits.astype(MLX::Core.float32), -1)
      )
      unis = MLX::Core.random_uniform([draft_tokens.size], 0.0, 1.0, MLX::Core.float32)
      log_unis = MLX::Core.log(
        MLX::Core.maximum(MLX::Core.subtract(unis, delta), 0.0)
      )
      accept_toks = MLX::Core.less_equal(
        log_unis,
        MLX::Core.subtract(model_probs, draft_probs)
      )
      (accept_toks.to_a + [false]).index(false)
    end

    def prefix_1d(array, length)
      return MLX::Core.array([], array.dtype) if length <= 0

      indices = MLX::Core.array((0...length).to_a, MLX::Core.int32)
      MLX::Core.take(array, indices, 0)
    end

    def tail_1d(array, count)
      return MLX::Core.array([], array.dtype) if count <= 0

      start = [array.shape[0] - count, 0].max
      indices = MLX::Core.array((start...array.shape[0]).to_a, MLX::Core.int32)
      MLX::Core.take(array, indices, 0)
    end

    def prefix_rows(array_2d, rows)
      return MLX::Core.array([], array_2d.dtype) if rows <= 0

      indices = MLX::Core.array((0...rows).to_a, MLX::Core.int32)
      MLX::Core.take(array_2d, indices, 0)
    end
  end
end
