# frozen_string_literal: true

require "optparse"

require_relative "lora"
require_relative "utils"

if $PROGRAM_NAME == __FILE__
  options = {
    model: "mlx_model",
    save_path: "lora_fused_model",
    adapter_file: "adapters.npz",
    synthetic_model: false,
    python_bin: ENV.fetch("PYTHON_BIN", "python3")
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby lora/fuse.rb [options]"
    opts.on("--model PATH", String, "Base model directory") { |v| options[:model] = v }
    opts.on("--save-path PATH", String, "Output fused model directory") { |v| options[:save_path] = v }
    opts.on("--adapter-file PATH", String, "Trained adapter weights (.npz)") { |v| options[:adapter_file] = v }
    opts.on("--synthetic-model", "Use synthetic model mode") { options[:synthetic_model] = true }
    opts.on("--python-bin BIN", String, "Python binary for tokenizer bridge") { |v| options[:python_bin] = v }
  end
  parser.parse!

  raise Errno::ENOENT, "Adapter file not found: #{options[:adapter_file]}" unless File.exist?(options[:adapter_file])

  model, tokenizer, config = LoraExample::Utils.load(
    options[:model],
    {},
    synthetic: options[:synthetic_model],
    python_bin: options[:python_bin]
  )

  adapters = MLX::Core.load(options[:adapter_file]).to_a
  lora_layers = adapters.count { |name, _| name.to_s.include?("q_proj.lora_a") }
  LoraExample::Train.inject_lora(model, lora_layers)

  model.load_weights(options[:adapter_file], strict: false)

  model.model.layers.each do |layer|
    if layer.self_attn.q_proj.is_a?(LoraExample::LoRALinear)
      layer.self_attn.q_proj = layer.self_attn.q_proj.to_linear
    end
    if layer.self_attn.v_proj.is_a?(LoraExample::LoRALinear)
      layer.self_attn.v_proj = layer.self_attn.v_proj.to_linear
    end
    if layer.respond_to?(:block_sparse_moe) && !layer.block_sparse_moe.nil? &&
       layer.block_sparse_moe.respond_to?(:gate) &&
       layer.block_sparse_moe.gate.is_a?(LoraExample::LoRALinear)
      layer.block_sparse_moe.gate = layer.block_sparse_moe.gate.to_linear
    end
  end

  weights = {}
  MLX::Utils.tree_flatten(model.parameters).each do |name, value|
    weights[name.to_s] = value
  end
  config = config.dup
  config.delete("quantization")

  LoraExample::Utils.save_model(options[:save_path], weights, tokenizer, config)
  puts "[INFO] Saved fused model to #{options[:save_path]}"
end
