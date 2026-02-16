# frozen_string_literal: true

require_relative "common"

module SegmentAnything
  class MaskEmbed < MLX::NN::Module
    def initialize(embed_dim:, mask_in_chans:)
      super()
      self.conv1 = MLX::NN::Conv2d.new(1, mask_in_chans / 4, 2, stride: 2)
      self.layer_norm1 = LayerNorm2d.new(mask_in_chans / 4)
      self.conv2 = MLX::NN::Conv2d.new(mask_in_chans / 4, mask_in_chans, 2, stride: 2)
      self.layer_norm2 = LayerNorm2d.new(mask_in_chans)
      self.conv3 = MLX::NN::Conv2d.new(mask_in_chans, embed_dim, 1)
      self.activation = MLX::NN::GELU.new
    end

    def call(x)
      x = activation.call(layer_norm1.call(conv1.call(x)))
      x = activation.call(layer_norm2.call(conv2.call(x)))
      conv3.call(x)
    end
  end

  class PositionEmbeddingRandom < MLX::NN::Module
    def initialize(num_pos_feats: 64, scale: nil)
      super()
      scale = 1.0 if scale.nil? || scale <= 0.0
      self.positional_embedding = MLX::Core.multiply(
        scale,
        MLX::Core.normal([2, num_pos_feats])
      )
    end

    def call(size)
      h, w = size
      grid = MLX::Core.ones([h, w], MLX::Core.float32)
      y_embed = MLX::Core.subtract(MLX::Core.cumsum(grid, 0), 0.5)
      x_embed = MLX::Core.subtract(MLX::Core.cumsum(grid, 1), 0.5)
      y_embed = MLX::Core.divide(y_embed, h.to_f)
      x_embed = MLX::Core.divide(x_embed, w.to_f)
      coords = MLX::Core.stack([x_embed, y_embed], -1)
      pe_encoding(coords)
    end

    def forward_with_coords(coords_input, image_size)
      h, w = image_size
      coords = coords_input.astype(MLX::Core.float32)
      scale = MLX::Core.array([w.to_f, h.to_f], MLX::Core.float32)
      pe_encoding(MLX::Core.divide(coords, scale))
    end

    private

    def pe_encoding(coords)
      base_shape = coords.shape[0...-1]
      n = base_shape.reduce(1, :*)
      coords = MLX::Core.subtract(MLX::Core.multiply(coords, 2.0), 1.0)
      flat = MLX::Core.reshape(coords, [n, 2])
      encoded = MLX::Core.matmul(flat, positional_embedding)
      encoded = MLX::Core.multiply(encoded, 2.0 * Math::PI)
      encoded = MLX::Core.concatenate([MLX::Core.sin(encoded), MLX::Core.cos(encoded)], 1)
      MLX::Core.reshape(encoded, base_shape + [encoded.shape[1]])
    end
  end

  class PromptEncoder < MLX::NN::Module
    attr_reader :embed_dim, :input_image_size, :image_embedding_size

    def initialize(
      embed_dim:,
      image_embedding_size:,
      input_image_size:,
      mask_in_chans:
    )
      super()
      @embed_dim = embed_dim
      @input_image_size = input_image_size
      @image_embedding_size = image_embedding_size
      @num_point_embeddings = 4

      self.point_embed = Array.new(@num_point_embeddings) { MLX::NN::Embedding.new(1, embed_dim) }
      self.not_a_point_embed = MLX::NN::Embedding.new(1, embed_dim)

      self.mask_input_size = [4 * image_embedding_size[0], 4 * image_embedding_size[1]]
      self.mask_embed = MaskEmbed.new(embed_dim: embed_dim, mask_in_chans: mask_in_chans)
      self.no_mask_embed = MLX::NN::Embedding.new(1, embed_dim)
    end

    def call(points:, boxes:, masks:, pe_layer:)
      bs = get_batch_size(points: points, boxes: boxes, masks: masks)
      sparse_embeddings = MLX::Core.zeros([bs, 0, embed_dim])

      unless points.nil?
        coords, labels = points
        point_embeddings = embed_points(coords, labels, pad: boxes.nil?, pe_layer: pe_layer)
        sparse_embeddings = MLX::Core.concatenate([sparse_embeddings, point_embeddings], 1)
      end

      unless boxes.nil?
        box_embeddings = embed_boxes(boxes, pe_layer: pe_layer)
        sparse_embeddings = MLX::Core.concatenate([sparse_embeddings, box_embeddings], 1)
      end

      dense_embeddings = if masks.nil?
        base = MLX::Core.reshape(no_mask_embed.weight, [1, 1, 1, embed_dim])
        MLX::Core.broadcast_to(base, [bs, image_embedding_size[0], image_embedding_size[1], embed_dim])
      else
        mask_embed.call(masks)
      end

      [sparse_embeddings, dense_embeddings]
    end

    private

    def get_batch_size(points:, boxes:, masks:)
      unless points.nil?
        return points[0].shape[0]
      end
      return boxes.shape[0] unless boxes.nil?
      return masks.shape[0] unless masks.nil?

      1
    end

    def embed_points(points, labels, pad:, pe_layer:)
      points = MLX::Core.add(points, 0.5)
      if pad
        padding_point = MLX::Core.zeros([points.shape[0], 1, 2])
        padding_label = MLX::Core.multiply(-1, MLX::Core.ones([labels.shape[0], 1]))
        points = MLX::Core.concatenate([points, padding_point], 1)
        labels = MLX::Core.concatenate([labels, padding_label], 1)
      end

      point_embedding = pe_layer.forward_with_coords(points, input_image_size)
      label_0 = MLX::Core.squeeze(point_embed[0].weight, 0)
      label_1 = MLX::Core.squeeze(point_embed[1].weight, 0)
      not_a_point = MLX::Core.squeeze(not_a_point_embed.weight, 0)

      merged = MLX::DSL::Tensor.where_labels(
        base: point_embedding,
        labels: labels,
        mapping: {
          0 => label_0,
          1 => label_1
        },
        mode: :add_or_replace
      )
      MLX::DSL::Tensor.where_labels(
        base: merged,
        labels: labels,
        mapping: {
          -1 => not_a_point
        },
        mode: :replace
      )
    end

    def embed_boxes(boxes, pe_layer:)
      boxes = MLX::Core.add(boxes, 0.5)
      coords = MLX::Core.reshape(boxes, [boxes.shape[0], 2, 2])
      corner_embedding = pe_layer.forward_with_coords(coords, input_image_size)
      add2 = MLX::Core.squeeze(point_embed[2].weight, 0)
      add3 = MLX::Core.squeeze(point_embed[3].weight, 0)
      corner_labels = MLX::Core.broadcast_to(
        MLX::Core.array([0, 1], MLX::Core.int32),
        [corner_embedding.shape[0], 2]
      )
      MLX::DSL::Tensor.where_labels(
        base: corner_embedding,
        labels: corner_labels,
        mapping: {
          0 => add2,
          1 => add3
        },
        mode: :add_or_replace
      )
    end
  end
end
