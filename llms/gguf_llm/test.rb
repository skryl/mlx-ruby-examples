# frozen_string_literal: true

require "optparse"

require_relative "models"

module GGUFLLM
  module TestHelpers
    module_function

    def token_id(token)
      value = token.to_a
      value = value[0] if value.is_a?(Array)
      value.to_i
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { seed: 13 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llms/gguf_llm/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])

  args = GGUFLLM::ModelArgs.new(
    hidden_size: 64,
    num_hidden_layers: 2,
    intermediate_size: 128,
    num_attention_heads: 4,
    rms_norm_eps: 1e-5,
    vocab_size: 101,
    context_length: 128
  )

  model = GGUFLLM::Model.new(args)
  prompt = MLX::Core.array([1, 2, 3], MLX::Core.int32)
  logits, cache = model.call(MLX::Core.expand_dims(prompt, 0))
  MLX::Core.eval(logits)
  unless logits.shape == [1, 3, args.vocab_size]
    raise "GGUF model forward shape mismatch: expected [1, 3, #{args.vocab_size}], got #{logits.shape.inspect}"
  end
  unless cache.length == args.num_hidden_layers
    raise "GGUF model cache length mismatch: expected #{args.num_hidden_layers}, got #{cache.length}"
  end

  generator = GGUFLLM.generate(prompt, model, temp: 0.0)
  token1 = generator.next
  MLX::Core.eval(token1)
  last_idx = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
  expected1_logits = MLX::Core.squeeze(MLX::Core.take(logits, last_idx, 1), 1)
  expected1 = MLX::Core.argmax(expected1_logits, -1)
  MLX::Core.eval(expected1)
  unless GGUFLLM::TestHelpers.token_id(token1) == GGUFLLM::TestHelpers.token_id(expected1)
    raise "First generated token mismatch against greedy forward argmax"
  end

  prompt2 = MLX::Core.array([1, 2, 3, GGUFLLM::TestHelpers.token_id(token1)], MLX::Core.int32)
  logits2, = model.call(MLX::Core.expand_dims(prompt2, 0))
  MLX::Core.eval(logits2)
  last_idx2 = MLX::Core.array([logits2.shape[1] - 1], MLX::Core.int32)
  expected2_logits = MLX::Core.squeeze(MLX::Core.take(logits2, last_idx2, 1), 1)
  expected2 = MLX::Core.argmax(expected2_logits, -1)
  MLX::Core.eval(expected2)

  token2 = generator.next
  MLX::Core.eval(token2)
  unless GGUFLLM::TestHelpers.token_id(token2) == GGUFLLM::TestHelpers.token_id(expected2)
    raise "Second generated token mismatch: cache path diverged from full forward"
  end

  translated = GGUFLLM.translate_weight_names("blk.0.attn_q.weight")
  unless translated == "model.layers.0.self_attn.q_proj.weight"
    raise "Weight translation mismatch: got #{translated}"
  end

  config = GGUFLLM.get_config(
    "llama.context_length" => 4096,
    "llama.embedding_length" => 1024,
    "llama.block_count" => 16,
    "llama.attention.head_count" => 8,
    "llama.feed_forward_length" => 4096,
    "llama.attention.head_count_kv" => 8,
    "llama.attention.layer_norm_rms_epsilon" => 1e-5,
    "tokenizer.ggml.tokens" => %w[a b c],
    "llama.rope.freq_base" => 10_000
  )
  unless config["vocab_size"] == 3 && config["rope_traditional"] == true
    raise "GGUF config extraction mismatch"
  end

  puts "Tests pass :)"
end
