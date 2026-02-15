# frozen_string_literal: true

require "json"
require "open3"
require "pathname"


require "mlx"

require_relative "language"
require_relative "vision"

module LlavaExample
  SCRIPT_DIR = Pathname.new(__dir__).join("python")

  class LlaVAConfig
    attr_reader :text_config,
                :vision_config,
                :ignore_index,
                :image_token_index,
                :vision_feature_select_strategy,
                :vision_feature_layer,
                :vocab_size

    def initialize(
      text_config:,
      vision_config:,
      ignore_index: -100,
      image_token_index: 32_000,
      vision_feature_select_strategy: "default",
      vision_feature_layer: -2,
      vocab_size: 32_000
    )
      @text_config = text_config
      @vision_config = vision_config
      @ignore_index = ignore_index
      @image_token_index = image_token_index
      @vision_feature_select_strategy = vision_feature_select_strategy
      @vision_feature_layer = vision_feature_layer
      @vocab_size = vocab_size
    end

    def self.from_dict(params)
      p = params.transform_keys(&:to_s)
      new(
        text_config: p.fetch("text_config"),
        vision_config: p.fetch("vision_config"),
        ignore_index: p.fetch("ignore_index", -100),
        image_token_index: p.fetch("image_token_index", 32_000),
        vision_feature_select_strategy: p.fetch("vision_feature_select_strategy", "default"),
        vision_feature_layer: p.fetch("vision_feature_layer", -2),
        vocab_size: p.fetch("vocab_size", 32_000)
      )
    end
  end

  class LlavaMultiModalProjector < MLX::NN::Module
    def initialize(config)
      super()
      self.linear_1 = MLX::NN::Linear.new(
        config.vision_config.hidden_size,
        config.text_config.hidden_size,
        bias: true
      )
      self.gelu = MLX::NN::GELU.new
      self.linear_2 = MLX::NN::Linear.new(
        config.text_config.hidden_size,
        config.text_config.hidden_size,
        bias: true
      )
    end

    def call(x)
      x = linear_1.call(x)
      x = gelu.call(x)
      linear_2.call(x)
    end
  end

  class LlavaModel < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config
      self.vision_tower = VisionModel.new(config.vision_config)
      self.language_model = LanguageModel.new(config.text_config)
      self.multi_modal_projector = LlavaMultiModalProjector.new(config)
      @vision_feature_layer = config.vision_feature_layer
      @vision_feature_select_strategy = config.vision_feature_select_strategy
    end

    def get_input_embeddings(input_ids:, pixel_values: nil)
      inputs_embeds = language_model.model.embed_tokens.call(input_ids)
      return inputs_embeds if pixel_values.nil?

      pixels = pixel_values
      if pixels.shape.length == 4 && pixels.shape[-1] != config.vision_config.num_channels && pixels.shape[1] == config.vision_config.num_channels
        pixels = MLX::Core.transpose(pixels, [0, 2, 3, 1])
      end

      _pooler_output, _vision_last_hidden, hidden_states = vision_tower.call(
        pixels,
        output_hidden_states: true
      )

      selected_image_feature = hidden_states[@vision_feature_layer]
      if @vision_feature_select_strategy == "default"
        selected_image_feature = MLX::Core.slice(
          selected_image_feature,
          [0, 1, 0],
          [
            selected_image_feature.shape[0],
            selected_image_feature.shape[1],
            selected_image_feature.shape[2]
          ]
        )
      elsif @vision_feature_select_strategy != "full"
        raise ArgumentError, "Unexpected feature selection strategy: #{@vision_feature_select_strategy}"
      end

      image_features = multi_modal_projector.call(selected_image_feature)
      _merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)
    end

    def _merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)
      image_token_index = config.image_token_index
      batch_size, num_image_patches, _embed_dim = image_features.shape
      unless batch_size == 1
        raise ArgumentError, "Only batch size 1 is currently supported for image feature merge"
      end

      ids = input_ids.to_a
      image_positions = []
      ids[0].each_with_index do |token_id, idx|
        image_positions << idx if token_id.to_i == image_token_index
      end

      if image_positions.length != num_image_patches
        raise ArgumentError,
              "The number of image tokens (#{image_positions.length}) does not match the number of image patches (#{num_image_patches})."
      end

      merged = inputs_embeds.to_a
      image_values = image_features.to_a
      image_positions.each_with_index do |pos, patch_idx|
        merged[0][pos] = image_values[0][patch_idx]
      end
      MLX::Core.array(merged, inputs_embeds.dtype)
    end

    def call(input_ids, pixel_values = nil, cache: nil)
      input_embeddings = get_input_embeddings(input_ids: input_ids, pixel_values: pixel_values)
      language_model.call(input_ids, cache: cache, inputs_embeds: input_embeddings)
    end

    def self.from_pretrained(path_or_hf_repo, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      path = Pathname.new(path_or_hf_repo.to_s)
      path = LlavaExample.snapshot_download(path_or_hf_repo.to_s, python_bin: python_bin) unless path.exist?

      config_path = path.join("config.json")
      raise Errno::ENOENT, "Could not find #{config_path}" unless config_path.exist?

      model_config = LlaVAConfig.from_dict(JSON.parse(File.binread(config_path)))
      text_cfg = model_config.text_config.is_a?(TextConfig) ? model_config.text_config : TextConfig.from_dict(model_config.text_config)
      vision_cfg = model_config.vision_config.is_a?(VisionConfig) ? model_config.vision_config : VisionConfig.from_dict(model_config.vision_config)
      config = LlaVAConfig.new(
        text_config: text_cfg,
        vision_config: vision_cfg,
        ignore_index: model_config.ignore_index,
        image_token_index: model_config.image_token_index,
        vision_feature_select_strategy: model_config.vision_feature_select_strategy,
        vision_feature_layer: model_config.vision_feature_layer,
        vocab_size: model_config.vocab_size
      )

      model = LlavaModel.new(config)

      weight_files = Dir.glob(path.join("*.safetensors").to_s).sort + Dir.glob(path.join("*.npz").to_s).sort
      if weight_files.empty?
        raise Errno::ENOENT, "No .safetensors/.npz weights found in #{path}"
      end

      weights = {}
      weight_files.each do |wf|
        MLX::Core.load(wf).to_a.each do |key, value|
          weights[key.to_s] = value
        end
      end

      weights = VisionModel.sanitize(weights)
      weights = LanguageModel.sanitize(weights)
      model.load_weights(weights.to_a, strict: false)
      model
    end
  end

  module_function

  def snapshot_download(repo_id, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    script_path = SCRIPT_DIR.join("snapshot_download.py").to_s
    stdout, stderr, status = Open3.capture3(python_bin, script_path, repo_id.to_s)
    unless status.success?
      raise RuntimeError, "Failed to download model snapshot for #{repo_id}: #{stderr}"
    end

    Pathname.new(stdout.strip)
  end
end
