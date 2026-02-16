# frozen_string_literal: true

require_relative "common"

module SegmentAnything
  module_function

  def upsample_nearest2d(x, scale: 2)
    batch, height, width, channels = x.shape

    x = MLX::Core.expand_dims(x, 2)
    x = MLX::Core.concatenate(Array.new(scale, x), 2)
    x = MLX::Core.reshape(x, [batch, height * scale, width, channels])

    x = MLX::Core.expand_dims(x, 3)
    x = MLX::Core.concatenate(Array.new(scale, x), 3)
    MLX::Core.reshape(x, [batch, height * scale, width * scale, channels])
  end

  class MLP < MLX::NN::Module
    def initialize(input_dim:, hidden_dim:, output_dim:, num_layers:, sigmoid_output: false)
      super()
      @num_layers = num_layers
      @sigmoid_output = sigmoid_output
      self.proj_in = MLX::NN::Linear.new(input_dim, hidden_dim)
      self.layers = Array.new(num_layers) { MLX::NN::Linear.new(hidden_dim, hidden_dim) }
      self.proj_out = MLX::NN::Linear.new(hidden_dim, output_dim)
    end

    def call(x)
      x = MLX::NN.relu(proj_in.call(x))
      layers.each do |layer|
        x = MLX::NN.relu(layer.call(x))
      end
      x = proj_out.call(x)
      return MLX::Core.sigmoid(x) if @sigmoid_output

      x
    end
  end

  class ConvTranspose2d < MLX::NN::Module
    def initialize(
      in_channels:,
      out_channels:,
      kernel_size:,
      stride: 1,
      padding: 0,
      bias: true
    )
      super()
      @stride = stride.is_a?(Array) ? stride[0] : stride
      self.conv = MLX::NN::Conv2d.new(
        in_channels,
        out_channels,
        kernel_size,
        padding: padding,
        bias: bias
      )
    end

    def call(x)
      up = if @stride.to_i > 1
        SegmentAnything.upsample_nearest2d(x, scale: @stride.to_i)
      else
        x
      end
      conv.call(up)
    end
  end

  class MaskDecoder < MLX::NN::Module
    def initialize(
      transformer_dim:,
      transformer:,
      num_multimask_outputs: 3,
      iou_head_depth: 3,
      iou_head_hidden_dim: 256
    )
      super()
      @transformer_dim = transformer_dim
      @num_multimask_outputs = num_multimask_outputs
      @num_mask_tokens = num_multimask_outputs + 1

      self.transformer = transformer
      self.iou_token = MLX::NN::Embedding.new(1, transformer_dim)
      self.mask_tokens = MLX::NN::Embedding.new(@num_mask_tokens, transformer_dim)

      self.upscale_conv1 = ConvTranspose2d.new(
        in_channels: transformer_dim,
        out_channels: transformer_dim / 4,
        kernel_size: 2,
        stride: 2,
        padding: 1
      )
      self.upscale_layer_norm = LayerNorm2d.new(transformer_dim / 4)
      self.activation = MLX::NN::GELU.new
      self.upscale_conv2 = ConvTranspose2d.new(
        in_channels: transformer_dim / 4,
        out_channels: transformer_dim / 8,
        kernel_size: 2,
        stride: 2,
        padding: 1
      )

      self.output_hypernetworks_mlps = Array.new(@num_mask_tokens) do
        MLP.new(
          input_dim: transformer_dim,
          hidden_dim: transformer_dim,
          output_dim: transformer_dim / 8,
          num_layers: 1
        )
      end

      self.iou_prediction_head = MLP.new(
        input_dim: transformer_dim,
        hidden_dim: iou_head_hidden_dim,
        output_dim: @num_mask_tokens,
        num_layers: [iou_head_depth - 2, 0].max
      )
    end

    def call(
      image_embeddings:,
      image_pe:,
      sparse_prompt_embeddings:,
      dense_prompt_embeddings:,
      multimask_output:
    )
      masks, iou_pred = predict_masks(
        image_embeddings: image_embeddings,
        image_pe: image_pe,
        sparse_prompt_embeddings: sparse_prompt_embeddings,
        dense_prompt_embeddings: dense_prompt_embeddings
      )

      if multimask_output
        idx = MLX::Core.array((1...@num_mask_tokens).to_a, MLX::Core.int32)
      else
        idx = MLX::Core.array([0], MLX::Core.int32)
      end
      masks = MLX::Core.take(masks, idx, 3)
      iou_pred = MLX::Core.take(iou_pred, idx, 1)
      [masks, iou_pred]
    end

    def predict_masks(
      image_embeddings:,
      image_pe:,
      sparse_prompt_embeddings:,
      dense_prompt_embeddings:
    )
      output_tokens = MLX::Core.concatenate([iou_token.weight, mask_tokens.weight], 0)
      bs = sparse_prompt_embeddings.shape[0]
      output_tokens = MLX::Core.broadcast_to(
        MLX::Core.expand_dims(output_tokens, 0),
        [bs, output_tokens.shape[0], output_tokens.shape[1]]
      )
      tokens = MLX::Core.concatenate([output_tokens, sparse_prompt_embeddings], 1)

      src = image_embeddings
      if src.shape[0] != tokens.shape[0]
        if src.shape[0] == 1
          src = MLX::Core.concatenate(Array.new(tokens.shape[0], src), 0)
        else
          raise ArgumentError, "Cannot broadcast image embeddings batch #{src.shape[0]} to #{tokens.shape[0]}"
        end
      end
      src = MLX::Core.add(src, dense_prompt_embeddings)
      b, h, w, c = src.shape

      hs, src_tokens = transformer.call(src, image_pe, tokens)
      iou_token_out = MLX::Core.squeeze(
        MLX::Core.take(hs, MLX::Core.array([0], MLX::Core.int32), 1),
        1
      )
      mask_tokens_out = MLX::Core.slice(hs, [0, 1, 0], [hs.shape[0], 1 + @num_mask_tokens, hs.shape[2]])

      src_tokens = MLX::Core.reshape(src_tokens, [b, h, w, c])
      src_tokens = upscale_conv1.call(src_tokens)
      src_tokens = activation.call(upscale_layer_norm.call(src_tokens))
      upscaled_embedding = activation.call(upscale_conv2.call(src_tokens))

      hyper_in = []
      @num_mask_tokens.times do |i|
        token_i = MLX::Core.squeeze(
          MLX::Core.take(mask_tokens_out, MLX::Core.array([i], MLX::Core.int32), 1),
          1
        )
        hyper_in << output_hypernetworks_mlps[i].call(token_i)
      end
      hyper_in = MLX::Core.stack(hyper_in, 1)

      bb, hh, ww, cc = upscaled_embedding.shape
      flat = MLX::Core.reshape(upscaled_embedding, [bb, hh * ww, cc])
      flat_t = MLX::Core.transpose(flat, [0, 2, 1])
      masks = MLX::Core.matmul(hyper_in, flat_t)
      masks = MLX::Core.transpose(masks, [0, 2, 1])
      masks = MLX::Core.reshape(masks, [bb, hh, ww, @num_mask_tokens])

      iou_pred = iou_prediction_head.call(iou_token_out)
      [masks, iou_pred]
    end
  end
end
