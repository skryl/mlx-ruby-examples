# frozen_string_literal: true

require "optparse"

require_relative "llava"
require_relative "../benchmark/parity"
module LlavaExample
  module TestHelpers
    module_function

    def token_id(token)
      value = token.to_a
      value = value[0] if value.is_a?(Array)
      value.to_i
    end

    def last_logits(logits)
      idx = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
      MLX::Core.squeeze(MLX::Core.take(logits, idx, 1), 1)
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { seed: 41 }
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby llava/test.rb [options]"
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!
  benchmark_enabled = ENV["MLX_BENCHMARK"] == "1"
  if benchmark_enabled
    BenchmarkParity.prime_backend!
    benchmark_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  MLX::Core.random_seed(options[:seed])

  text_config = LlavaExample::TextConfig.new(
    model_type: "llama",
    hidden_size: 64,
    num_hidden_layers: 2,
    intermediate_size: 128,
    num_attention_heads: 4,
    rms_norm_eps: 1e-5,
    vocab_size: 257,
    num_key_value_heads: 4
  )
  vision_config = LlavaExample::VisionConfig.new(
    model_type: "clip_vision_model",
    num_hidden_layers: 2,
    hidden_size: 64,
    intermediate_size: 128,
    num_attention_heads: 4,
    image_size: 16,
    patch_size: 8,
    num_channels: 3,
    layer_norm_eps: 1e-5
  )
  config = LlavaExample::LlaVAConfig.new(
    text_config: text_config,
    vision_config: vision_config,
    image_token_index: 256,
    vision_feature_select_strategy: "default",
    vision_feature_layer: -2,
    vocab_size: text_config.vocab_size
  )

  model = LlavaExample::LlavaModel.new(config)
  BenchmarkDeterministic.reinitialize_module!(model) if benchmark_enabled

  input_ids = MLX::Core.array([[1, 256, 256, 256, 256, 7, 8, 9]], MLX::Core.int32)
  pixel_values = MLX::Core.random_uniform([1, 3, 16, 16], 0.0, 1.0, MLX::Core.float32)
  logits, cache = model.call(input_ids, pixel_values)
  MLX::Core.eval(logits)
  unless logits.shape == [1, input_ids.shape[1], text_config.vocab_size]
    raise "LLaVA forward shape mismatch: expected [1, #{input_ids.shape[1]}, #{text_config.vocab_size}], got #{logits.shape.inspect}"
  end
  unless cache.length == text_config.num_hidden_layers
    raise "LLaVA cache length mismatch: expected #{text_config.num_hidden_layers}, got #{cache.length}"
  end

  if benchmark_enabled
    if ENV["MLX_BENCHMARK_DRYRUN"] == "1"
      exit 0
    end
    benchmark_parity_started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    BenchmarkParity.validate!(
      model_id: "llava",
      inputs: { input_ids: input_ids, pixel_values: pixel_values },
      outputs: { logits: logits },
      python_bin: ENV.fetch("PYTHON_BIN", "python3")
    )
    benchmark_parity_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_parity_started_at
    benchmark_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - benchmark_started_at - benchmark_parity_elapsed
    puts format("BENCHMARK_SECONDS=%.9f", benchmark_elapsed)
    puts "Tests pass :)"
    exit 0
  end

  inputs_embeds = model.language_model.model.embed_tokens.call(input_ids)
  vision_input = MLX::Core.transpose(pixel_values, [0, 2, 3, 1])
  _pool, _last, hidden_states = model.vision_tower.call(vision_input, output_hidden_states: true)
  selected = hidden_states[config.vision_feature_layer]
  selected = MLX::Core.slice(
    selected,
    [0, 1, 0],
    [selected.shape[0], selected.shape[1], selected.shape[2]]
  )
  image_features = model.multi_modal_projector.call(selected)

  merged = model._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)
  MLX::Core.eval(merged)

  image_positions = [1, 2, 3, 4]
  image_positions.each_with_index do |position, patch_idx|
    pos_idx = MLX::Core.array([position], MLX::Core.int32)
    patch_index = MLX::Core.array([patch_idx], MLX::Core.int32)
    merged_token = MLX::Core.squeeze(MLX::Core.take(merged, pos_idx, 1), 1)
    patch_token = MLX::Core.squeeze(MLX::Core.take(image_features, patch_index, 1), 1)
    delta = MLX::Core.sum(MLX::Core.abs(MLX::Core.subtract(merged_token, patch_token)))
    MLX::Core.eval(delta)
    if delta.item > 1e-5
      raise "Image feature merge mismatch at sequence position #{position}: delta=#{delta.item}"
    end
  end

  original_first = MLX::Core.squeeze(MLX::Core.take(inputs_embeds, MLX::Core.array([0], MLX::Core.int32), 1), 1)
  merged_first = MLX::Core.squeeze(MLX::Core.take(merged, MLX::Core.array([0], MLX::Core.int32), 1), 1)
  delta_first = MLX::Core.sum(MLX::Core.abs(MLX::Core.subtract(original_first, merged_first)))
  MLX::Core.eval(delta_first)
  if delta_first.item > 1e-5
    raise "Merge changed non-image token embeddings unexpectedly"
  end

  bad_ids = MLX::Core.array([[1, 256, 256, 7, 8, 9, 10, 11]], MLX::Core.int32)
  bad_embeds = model.language_model.model.embed_tokens.call(bad_ids)
  begin
    model._merge_input_ids_with_image_features(image_features, bad_embeds, bad_ids)
    raise "Expected merge mismatch to raise an error"
  rescue StandardError => e
    unless e.message.include?("number of image tokens")
      raise "Unexpected merge error message: #{e.message}"
    end
  end

  logits1, cache1 = model.call(input_ids, pixel_values)
  next_token = MLX::Core.argmax(LlavaExample::TestHelpers.last_logits(logits1), -1)
  MLX::Core.eval(next_token)

  step_logits, step_cache = model.language_model.call(MLX::Core.expand_dims(next_token, 0), cache: cache1)
  second_token_cached = MLX::Core.argmax(LlavaExample::TestHelpers.last_logits(step_logits), -1)
  MLX::Core.eval(second_token_cached)

  full_ids = MLX::Core.concatenate([input_ids, MLX::Core.expand_dims(next_token, 1)], 1)
  full_logits, _full_cache = model.call(full_ids, pixel_values)
  second_token_full = MLX::Core.argmax(LlavaExample::TestHelpers.last_logits(full_logits), -1)
  MLX::Core.eval(second_token_full)

  unless LlavaExample::TestHelpers.token_id(second_token_cached) == LlavaExample::TestHelpers.token_id(second_token_full)
    raise "Cached decode mismatch: cached=#{LlavaExample::TestHelpers.token_id(second_token_cached)}, full=#{LlavaExample::TestHelpers.token_id(second_token_full)}"
  end

  unless step_cache.length == text_config.num_hidden_layers
    raise "Language-only step cache length mismatch"
  end

  puts "Tests pass :)"
end
