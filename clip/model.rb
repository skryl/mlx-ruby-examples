# frozen_string_literal: true


require "mlx"

module ClipExample
  class CLIPModel < MLX::NN::Module
    def initialize(
      vocab_size: 259,
      text_width: 256,
      vision_width: 256,
      embed_dim: 128,
      image_size: 224,
      patch_size: 16,
      max_length: 77
    )
      super()
      @embed_dim = embed_dim
      @image_size = image_size
      @patch_size = patch_size
      @max_length = max_length

      self.token_embedding = MLX::NN::Embedding.new(vocab_size, text_width)
      self.text_projection = MLX::NN::Linear.new(text_width, embed_dim, bias: false)

      self.vision_conv = MLX::NN::Conv2d.new(3, vision_width, patch_size, stride: patch_size, bias: false)
      self.vision_projection = MLX::NN::Linear.new(vision_width, embed_dim, bias: false)

      self.logit_scale = MLX::Core.array(Math.log(1.0 / 0.07), MLX::Core.float32)
    end

    def encode_text(input_ids)
      x = token_embedding.call(input_ids)
      x = MLX::Core.mean(x, 1)
      x = text_projection.call(x)
      normalize(x)
    end

    def encode_image(pixel_values)
      x = pixel_values
      x = MLX::Core.expand_dims(x, 0) if x.ndim == 3
      x = vision_conv.call(x)
      x = MLX::Core.mean(x, 1)
      x = MLX::Core.mean(x, 1)
      x = vision_projection.call(x)
      normalize(x)
    end

    def call(input_ids:, pixel_values:, return_loss: false)
      text_embeds = encode_text(input_ids)
      image_embeds = encode_image(pixel_values)

      scale = MLX::Core.exp(logit_scale)
      logits_per_image = MLX::Core.multiply(
        scale,
        MLX::Core.matmul(image_embeds, MLX::Core.transpose(text_embeds, [1, 0]))
      )
      logits_per_text = MLX::Core.transpose(logits_per_image, [1, 0])

      if return_loss
        labels = MLX::Core.arange(0, logits_per_image.shape[0], 1).astype(MLX::Core.int32)
        loss_i = MLX::Core.mean(MLX::NN::Losses.cross_entropy(logits_per_image, labels))
        loss_t = MLX::Core.mean(MLX::NN::Losses.cross_entropy(logits_per_text, labels))
        loss = MLX::Core.multiply(0.5, MLX::Core.add(loss_i, loss_t))
      else
        loss = nil
      end

      {
        "text_embeds" => text_embeds,
        "image_embeds" => image_embeds,
        "logits_per_image" => logits_per_image,
        "logits_per_text" => logits_per_text,
        "loss" => loss
      }
    end

    private

    def normalize(x)
      norm = MLX::Core.sqrt(MLX::Core.sum(MLX::Core.square(x), -1))
      norm = MLX::Core.expand_dims(norm, 1)
      MLX::Core.divide(x, MLX::Core.maximum(norm, 1e-6))
    end
  end
end
