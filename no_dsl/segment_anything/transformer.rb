# frozen_string_literal: true

require_relative "common"

module SegmentAnything
  class Attention < MLX::NN::Module
    def initialize(embedding_dim:, num_heads:, downsample_rate: 1)
      super()
      @embedding_dim = embedding_dim
      @internal_dim = embedding_dim / downsample_rate
      @num_heads = num_heads
      unless (@internal_dim % @num_heads).zero?
        raise ArgumentError, "num_heads must divide embedding_dim"
      end

      self.q_proj = MLX::NN::Linear.new(embedding_dim, @internal_dim)
      self.k_proj = MLX::NN::Linear.new(embedding_dim, @internal_dim)
      self.v_proj = MLX::NN::Linear.new(embedding_dim, @internal_dim)
      self.out_proj = MLX::NN::Linear.new(@internal_dim, embedding_dim)
    end

    def call(q:, k:, v:)
      q = separate_heads(q_proj.call(q))
      k = separate_heads(k_proj.call(k))
      v = separate_heads(v_proj.call(v))

      c_per_head = q.shape[3]
      scores = MLX::Core.matmul(q, MLX::Core.transpose(k, [0, 1, 3, 2]))
      scores = MLX::Core.divide(scores, Math.sqrt(c_per_head.to_f))
      attn = MLX::Core.softmax(scores, -1)

      out = MLX::Core.matmul(attn, v)
      out = recombine_heads(out)
      out_proj.call(out)
    end

    private

    def separate_heads(x)
      b, n, c = x.shape
      per_head = c / @num_heads
      x = MLX::Core.reshape(x, [b, n, @num_heads, per_head])
      MLX::Core.transpose(x, [0, 2, 1, 3])
    end

    def recombine_heads(x)
      b, n_heads, n_tokens, c_per_head = x.shape
      x = MLX::Core.transpose(x, [0, 2, 1, 3])
      MLX::Core.reshape(x, [b, n_tokens, n_heads * c_per_head])
    end
  end

  class TwoWayAttentionBlock < MLX::NN::Module
    def initialize(
      embedding_dim:,
      num_heads:,
      mlp_dim: 2048,
      attention_downsample_rate: 2,
      skip_first_layer_pe: false
    )
      super()
      self.self_attn = Attention.new(embedding_dim: embedding_dim, num_heads: num_heads)
      self.layer_norm1 = MLX::NN::LayerNorm.new(embedding_dim)

      self.cross_attn_token_to_image = Attention.new(
        embedding_dim: embedding_dim,
        num_heads: num_heads,
        downsample_rate: attention_downsample_rate
      )
      self.layer_norm2 = MLX::NN::LayerNorm.new(embedding_dim)

      self.mlp = MLPBlock.new(embedding_dim: embedding_dim, mlp_dim: mlp_dim)
      self.layer_norm3 = MLX::NN::LayerNorm.new(embedding_dim)

      self.layer_norm4 = MLX::NN::LayerNorm.new(embedding_dim)
      self.cross_attn_image_to_token = Attention.new(
        embedding_dim: embedding_dim,
        num_heads: num_heads,
        downsample_rate: attention_downsample_rate
      )
      @skip_first_layer_pe = skip_first_layer_pe
    end

    def call(queries:, keys:, query_pe:, key_pe:)
      if @skip_first_layer_pe
        queries = self_attn.call(q: queries, k: queries, v: queries)
      else
        q = MLX::Core.add(queries, query_pe)
        attn_out = self_attn.call(q: q, k: q, v: queries)
        queries = MLX::Core.add(queries, attn_out)
      end
      queries = layer_norm1.call(queries)

      q = MLX::Core.add(queries, query_pe)
      k = MLX::Core.add(keys, key_pe)
      attn_out = cross_attn_token_to_image.call(q: q, k: k, v: keys)
      queries = layer_norm2.call(MLX::Core.add(queries, attn_out))

      mlp_out = mlp.call(queries)
      queries = layer_norm3.call(MLX::Core.add(queries, mlp_out))

      q2 = MLX::Core.add(queries, query_pe)
      k2 = MLX::Core.add(keys, key_pe)
      attn_out2 = cross_attn_image_to_token.call(q: k2, k: q2, v: queries)
      keys = layer_norm4.call(MLX::Core.add(keys, attn_out2))

      [queries, keys]
    end
  end

  class TwoWayTransformer < MLX::NN::Module
    def initialize(
      depth:,
      embedding_dim:,
      num_heads:,
      mlp_dim:,
      attention_downsample_rate: 2
    )
      super()
      self.layers = Array.new(depth) do |i|
        TwoWayAttentionBlock.new(
          embedding_dim: embedding_dim,
          num_heads: num_heads,
          mlp_dim: mlp_dim,
          attention_downsample_rate: attention_downsample_rate,
          skip_first_layer_pe: i.zero?
        )
      end

      self.final_attn_token_to_image = Attention.new(
        embedding_dim: embedding_dim,
        num_heads: num_heads,
        downsample_rate: attention_downsample_rate
      )
      self.layer_norm_final_attn = MLX::NN::LayerNorm.new(embedding_dim)
    end

    def call(image_embedding, image_pe, point_embedding)
      batch_size, h, w, c = image_embedding.shape
      image_embedding = MLX::Core.reshape(image_embedding, [batch_size, h * w, c])
      image_pe = MLX::Core.reshape(image_pe, [h * w, c])

      queries = point_embedding
      keys = image_embedding

      layers.each do |layer|
        queries, keys = layer.call(
          queries: queries,
          keys: keys,
          query_pe: point_embedding,
          key_pe: image_pe
        )
      end

      q = MLX::Core.add(queries, point_embedding)
      k = MLX::Core.add(keys, image_pe)
      attn_out = final_attn_token_to_image.call(q: q, k: k, v: keys)
      queries = layer_norm_final_attn.call(MLX::Core.add(queries, attn_out))

      [queries, keys]
    end
  end
end
