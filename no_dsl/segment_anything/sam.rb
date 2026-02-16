# frozen_string_literal: true

require "json"
require "pathname"

require_relative "image_encoder"
require_relative "mask_decoder"
require_relative "prompt_encoder"
require_relative "transformer"

module SegmentAnything
  module_function

  def resize_nearest_hw(x, out_h, out_w)
    in_h = x.shape[1]
    in_w = x.shape[2]

    h_idx = Array.new(out_h) { |i| [(i.to_f * in_h / out_h).floor, in_h - 1].min }
    w_idx = Array.new(out_w) { |i| [(i.to_f * in_w / out_w).floor, in_w - 1].min }

    h_idx = MLX::Core.array(h_idx, MLX::Core.int32)
    w_idx = MLX::Core.array(w_idx, MLX::Core.int32)
    out = MLX::Core.take(x, h_idx, 1)
    MLX::Core.take(out, w_idx, 2)
  end

  class Sam < MLX::NN::Module
    attr_reader :mask_threshold, :image_format

    def initialize(
      vision_encoder:,
      prompt_encoder:,
      mask_decoder:,
      pixel_mean: [123.675, 116.28, 103.53],
      pixel_std: [58.395, 57.12, 57.375]
    )
      super()
      @mask_threshold = 0.0
      @image_format = "RGB"

      self.vision_encoder = vision_encoder
      self.prompt_encoder = prompt_encoder
      self.mask_decoder = mask_decoder

      self.pixel_mean = MLX::Core.reshape(MLX::Core.array(pixel_mean, MLX::Core.float32), [1, 1, 1, 3])
      self.pixel_std = MLX::Core.reshape(MLX::Core.array(pixel_std, MLX::Core.float32), [1, 1, 1, 3])
      self.shared_image_embedding = PositionEmbeddingRandom.new(num_pos_feats: prompt_encoder.embed_dim / 2)
    end

    def call(batched_input, multimask_output: true)
      input_images = batched_input.map do |record|
        preprocess(fetch_key(record, :image))
      end
      input_images = MLX::Core.stack(input_images, 0)
      image_embeddings = vision_encoder.call(input_images)

      outputs = []
      batched_input.each_with_index do |record, idx|
        curr_embedding = MLX::Core.take(image_embeddings, MLX::Core.array([idx], MLX::Core.int32), 0)
        curr_embedding = MLX::Core.squeeze(curr_embedding, 0)

        points = nil
        point_coords = fetch_key(record, :point_coords)
        point_labels = fetch_key(record, :point_labels)
        points = [point_coords, point_labels] unless point_coords.nil?

        sparse_embeddings, dense_embeddings = prompt_encoder.call(
          points: points,
          boxes: fetch_key(record, :boxes),
          masks: fetch_key(record, :mask_inputs),
          pe_layer: shared_image_embedding
        )

        low_res_masks, iou_predictions = mask_decoder.call(
          image_embeddings: MLX::Core.expand_dims(curr_embedding, 0),
          image_pe: shared_image_embedding.call(prompt_encoder.image_embedding_size),
          sparse_prompt_embeddings: sparse_embeddings,
          dense_prompt_embeddings: dense_embeddings,
          multimask_output: multimask_output
        )

        image = fetch_key(record, :image)
        input_size = [image.shape[0], image.shape[1]]
        original_size = fetch_key(record, :original_size) || input_size

        masks = postprocess_masks(
          low_res_masks,
          input_size: input_size,
          original_size: original_size
        )
        masks = MLX::Core.greater(masks, mask_threshold)

        outputs << {
          "masks" => masks,
          "iou_predictions" => iou_predictions,
          "low_res_logits" => low_res_masks
        }
      end

      outputs
    end

    def postprocess_masks(masks, input_size:, original_size:)
      masks = SegmentAnything.resize_nearest_hw(masks, vision_encoder.img_size, vision_encoder.img_size)
      masks = MLX::Core.slice(masks, [0, 0, 0, 0], [masks.shape[0], input_size[0], input_size[1], masks.shape[3]])
      SegmentAnything.resize_nearest_hw(masks, original_size[0], original_size[1])
    end

    def preprocess(x)
      single = x.shape.length == 3
      x = MLX::Core.expand_dims(x, 0) if single
      x = MLX::Core.divide(MLX::Core.subtract(x.astype(MLX::Core.float32), pixel_mean), pixel_std)

      h = x.shape[1]
      w = x.shape[2]
      padh = vision_encoder.img_size - h
      padw = vision_encoder.img_size - w

      if padh > 0
        pad_h = MLX::Core.zeros([x.shape[0], padh, w, x.shape[3]], x.dtype)
        x = MLX::Core.concatenate([x, pad_h], 1)
      end
      if padw > 0
        pad_w = MLX::Core.zeros([x.shape[0], x.shape[1], padw, x.shape[3]], x.dtype)
        x = MLX::Core.concatenate([x, pad_w], 2)
      end

      single ? MLX::Core.squeeze(x, 0) : x
    end

    private

    def fetch_key(hash, key)
      hash[key] || hash[key.to_s]
    end
  end

  def load(model_path)
    model_path = Pathname.new(model_path)
    config_path = model_path.join("config.json")
    raise Errno::ENOENT, "Missing config file: #{config_path}" unless config_path.exist?

    config = JSON.parse(File.binread(config_path))
    vision = config.fetch("vision_config")

    encoder_embed_dim = vision.fetch("hidden_size")
    encoder_depth = vision.fetch("num_hidden_layers")
    encoder_num_heads = vision.fetch("num_attention_heads")
    encoder_global_attn_indexes = vision.fetch("global_attn_indexes", [])

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size / vit_patch_size

    sam_model = Sam.new(
      vision_encoder: ImageEncoderViT.new(
        depth: encoder_depth,
        embed_dim: encoder_embed_dim,
        img_size: image_size,
        mlp_ratio: 4.0,
        num_heads: encoder_num_heads,
        patch_size: vit_patch_size,
        qkv_bias: true,
        global_attn_indexes: encoder_global_attn_indexes,
        window_size: 14,
        out_chans: prompt_embed_dim
      ),
      prompt_encoder: PromptEncoder.new(
        embed_dim: prompt_embed_dim,
        image_embedding_size: [image_embedding_size, image_embedding_size],
        input_image_size: [image_size, image_size],
        mask_in_chans: 16
      ),
      mask_decoder: MaskDecoder.new(
        num_multimask_outputs: 3,
        transformer: TwoWayTransformer.new(
          depth: 2,
          embedding_dim: prompt_embed_dim,
          mlp_dim: 2048,
          num_heads: 8
        ),
        transformer_dim: prompt_embed_dim,
        iou_head_depth: 3,
        iou_head_hidden_dim: 256
      )
    )

    weight_file = model_path.join("model.safetensors")
    sam_model.load_weights(weight_file.to_s, strict: false) if weight_file.exist?
    sam_model
  end

  def build_tiny_model(
    image_size: 64,
    patch_size: 8,
    embed_dim: 128,
    depth: 2,
    num_heads: 4,
    prompt_embed_dim: 64
  )
    image_embedding_size = image_size / patch_size
    Sam.new(
      vision_encoder: ImageEncoderViT.new(
        depth: depth,
        embed_dim: embed_dim,
        img_size: image_size,
        mlp_ratio: 2.0,
        num_heads: num_heads,
        patch_size: patch_size,
        qkv_bias: true,
        out_chans: prompt_embed_dim
      ),
      prompt_encoder: PromptEncoder.new(
        embed_dim: prompt_embed_dim,
        image_embedding_size: [image_embedding_size, image_embedding_size],
        input_image_size: [image_size, image_size],
        mask_in_chans: 16
      ),
      mask_decoder: MaskDecoder.new(
        num_multimask_outputs: 3,
        transformer: TwoWayTransformer.new(
          depth: 2,
          embedding_dim: prompt_embed_dim,
          mlp_dim: 256,
          num_heads: 4
        ),
        transformer_dim: prompt_embed_dim,
        iou_head_depth: 3,
        iou_head_hidden_dim: 64
      )
    )
  end
end
