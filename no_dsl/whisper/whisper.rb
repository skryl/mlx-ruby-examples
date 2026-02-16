# frozen_string_literal: true

require "base64"
require "zlib"

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module WhisperExample
  class ModelDimensions
    attr_reader :n_mels,
                :n_audio_ctx,
                :n_audio_state,
                :n_audio_head,
                :n_audio_layer,
                :n_vocab,
                :n_text_ctx,
                :n_text_state,
                :n_text_head,
                :n_text_layer

    def initialize(
      n_mels: 80,
      n_audio_ctx: 1500,
      n_audio_state: 384,
      n_audio_head: 6,
      n_audio_layer: 4,
      n_vocab: 51_865,
      n_text_ctx: 448,
      n_text_state: 384,
      n_text_head: 6,
      n_text_layer: 4
    )
      @n_mels = n_mels
      @n_audio_ctx = n_audio_ctx
      @n_audio_state = n_audio_state
      @n_audio_head = n_audio_head
      @n_audio_layer = n_audio_layer
      @n_vocab = n_vocab
      @n_text_ctx = n_text_ctx
      @n_text_state = n_text_state
      @n_text_head = n_text_head
      @n_text_layer = n_text_layer
    end

    def self.from_hash(raw)
      data = raw.transform_keys(&:to_s)
      new(
        n_mels: data.fetch("n_mels", 80),
        n_audio_ctx: data.fetch("n_audio_ctx", 1500),
        n_audio_state: data.fetch("n_audio_state", 384),
        n_audio_head: data.fetch("n_audio_head", 6),
        n_audio_layer: data.fetch("n_audio_layer", 4),
        n_vocab: data.fetch("n_vocab", 51_865),
        n_text_ctx: data.fetch("n_text_ctx", 448),
        n_text_state: data.fetch("n_text_state", 384),
        n_text_head: data.fetch("n_text_head", 6),
        n_text_layer: data.fetch("n_text_layer", 4)
      )
    end

    def to_h
      {
        "n_mels" => n_mels,
        "n_audio_ctx" => n_audio_ctx,
        "n_audio_state" => n_audio_state,
        "n_audio_head" => n_audio_head,
        "n_audio_layer" => n_audio_layer,
        "n_vocab" => n_vocab,
        "n_text_ctx" => n_text_ctx,
        "n_text_state" => n_text_state,
        "n_text_head" => n_text_head,
        "n_text_layer" => n_text_layer
      }
    end
  end

  module ModelOps
    module_function

    def sinusoids(length, channels, max_timescale: 10_000.0)
      raise ArgumentError, "channels must be even" unless (channels % 2).zero?

      half = channels / 2
      log_inc = Math.log(max_timescale) / [half - 1, 1].max
      inv = MLX::Core.exp(MLX::Core.multiply(-log_inc, MLX::Core.arange(0, half, 1, MLX::Core.float32)))
      scaled = MLX::Core.multiply(
        MLX::Core.expand_dims(MLX::Core.arange(0, length, 1, MLX::Core.float32), 1),
        MLX::Core.expand_dims(inv, 0)
      )
      MLX::Core.concatenate([MLX::Core.sin(scaled), MLX::Core.cos(scaled)], 1)
    end

    def create_additive_causal_mask(n)
      row = MLX::Core.expand_dims(MLX::Core.arange(0, n, 1), 1)
      col = MLX::Core.expand_dims(MLX::Core.arange(0, n, 1), 0)
      mask = MLX::Core.less(row, col).astype(MLX::Core.float32)
      MLX::Core.multiply(mask, -1e9)
    end

    def slice_time(x, start_idx, end_idx)
      MLX::Core.slice(x, [0, start_idx, 0], [x.shape[0], end_idx, x.shape[2]])
    end

    def downsample_2x(x)
      idx = MLX::Core.array((0...x.shape[1]).step(2).to_a, MLX::Core.int32)
      MLX::Core.take(x, idx, 1)
    end
  end

  class MultiHeadAttention < MLX::NN::Module
    def initialize(n_state, n_head)
      super()
      @n_head = n_head
      self.query = MLX::NN::Linear.new(n_state, n_state)
      self.key = MLX::NN::Linear.new(n_state, n_state, bias: false)
      self.value = MLX::NN::Linear.new(n_state, n_state)
      self.out = MLX::NN::Linear.new(n_state, n_state)
    end

    def call(x, xa: nil, mask: nil, kv_cache: nil)
      q = query.call(x)

      if xa.nil?
        k = key.call(x)
        v = value.call(x)
        unless kv_cache.nil?
          k = MLX::Core.concatenate([kv_cache[0], k], 1)
          v = MLX::Core.concatenate([kv_cache[1], v], 1)
        end
      elsif kv_cache.nil?
        k = key.call(xa)
        v = value.call(xa)
      else
        k, v = kv_cache
      end

      wv, qk = qkv_attention(q, k, v, mask)
      [out.call(wv), [k, v], qk]
    end

    def qkv_attention(q, k, v, mask = nil)
      n_batch, n_ctx, n_state = q.shape
      n_k = k.shape[1]
      head_dim = n_state / @n_head
      scale = head_dim**-0.5

      q = MLX::Core.transpose(MLX::Core.reshape(q, [n_batch, n_ctx, @n_head, head_dim]), [0, 2, 1, 3])
      k = MLX::Core.transpose(MLX::Core.reshape(k, [n_batch, n_k, @n_head, head_dim]), [0, 2, 3, 1])
      v = MLX::Core.transpose(MLX::Core.reshape(v, [n_batch, n_k, @n_head, head_dim]), [0, 2, 1, 3])

      qk = MLX::Core.multiply(MLX::Core.matmul(q, k), scale)
      unless mask.nil?
        m = MLX::Core.slice(mask, [0, 0], [n_ctx, n_k])
        qk = MLX::Core.add(qk, m)
      end

      w = MLX::Core.softmax(qk.astype(MLX::Core.float32), -1).astype(qk.dtype)
      out = MLX::Core.transpose(MLX::Core.matmul(w, v), [0, 2, 1, 3])
      out = MLX::Core.reshape(out, [n_batch, n_ctx, n_state])
      [out, qk]
    end
  end

  class ResidualAttentionBlock < MLX::NN::Module
    def initialize(n_state, n_head, cross_attention: false)
      super()

      self.attn = MultiHeadAttention.new(n_state, n_head)
      self.attn_ln = MLX::NN::LayerNorm.new(n_state)

      self.cross_attn = cross_attention ? MultiHeadAttention.new(n_state, n_head) : nil
      self.cross_attn_ln = cross_attention ? MLX::NN::LayerNorm.new(n_state) : nil

      n_mlp = n_state * 4
      self.mlp1 = MLX::NN::Linear.new(n_state, n_mlp)
      self.mlp2 = MLX::NN::Linear.new(n_mlp, n_state)
      self.mlp_ln = MLX::NN::LayerNorm.new(n_state)
    end

    def call(x, xa: nil, mask: nil, kv_cache: nil)
      kv, cross_kv = kv_cache || [nil, nil]
      y, kv, _qk = attn.call(attn_ln.call(x), mask: mask, kv_cache: kv)
      x = MLX::Core.add(x, y)

      cross_qk = nil
      unless cross_attn.nil?
        y, cross_kv, cross_qk = cross_attn.call(cross_attn_ln.call(x), xa: xa, kv_cache: cross_kv)
        x = MLX::Core.add(x, y)
      end

      x = MLX::Core.add(x, mlp2.call(MLX::NN.gelu(mlp1.call(mlp_ln.call(x)))))
      [x, [kv, cross_kv], cross_qk]
    end
  end

  class AudioEncoder < MLX::NN::Module
    def initialize(n_mels, n_ctx, n_state, n_head, n_layer, dtype: MLX::Core.float16)
      super()
      @n_ctx = n_ctx
      self.conv1 = MLX::NN::Linear.new(n_mels, n_state)
      self.conv2 = MLX::NN::Linear.new(n_state, n_state)
      self.positional_embedding = ModelOps.sinusoids(n_ctx, n_state).astype(dtype)
      self.blocks = Array.new(n_layer) { ResidualAttentionBlock.new(n_state, n_head) }
      self.ln_post = MLX::NN::LayerNorm.new(n_state)
    end

    def call(x)
      h = MLX::NN.gelu(conv1.call(x))
      h = ModelOps.downsample_2x(h)
      h = MLX::NN.gelu(conv2.call(h))

      if h.shape[1] > @n_ctx
        h = ModelOps.slice_time(h, 0, @n_ctx)
      elsif h.shape[1] < @n_ctx
        h = MLX::Core.pad(h, [[0, 0], [0, @n_ctx - h.shape[1]], [0, 0]])
      end

      h = MLX::Core.add(h, MLX::Core.expand_dims(positional_embedding, 0))
      blocks.each { |block| h, = block.call(h) }
      ln_post.call(h)
    end
  end

  class TextDecoder < MLX::NN::Module
    def initialize(n_vocab, n_ctx, n_state, n_head, n_layer, dtype: MLX::Core.float16)
      super()
      @n_ctx = n_ctx
      self.token_embedding = MLX::NN::Embedding.new(n_vocab, n_state)
      self.positional_embedding = MLX::Core.zeros([n_ctx, n_state], dtype)
      self.blocks = Array.new(n_layer) { ResidualAttentionBlock.new(n_state, n_head, cross_attention: true) }
      self.ln = MLX::NN::LayerNorm.new(n_state)
      self.output_proj = MLX::NN::Linear.new(n_state, n_vocab, bias: false)
      self.mask = ModelOps.create_additive_causal_mask(n_ctx).astype(dtype)
    end

    def call(x, xa, kv_cache: nil)
      validate_token_ids!(x)

      offset = if kv_cache && !kv_cache.empty? && !kv_cache[0].nil? && !kv_cache[0][0].nil? && !kv_cache[0][0][0].nil?
                 kv_cache[0][0][0].shape[1]
               else
                 0
               end

      seq_len = x.shape[1]
      pos = MLX::Core.slice(positional_embedding, [offset, 0], [offset + seq_len, positional_embedding.shape[1]])
      h = MLX::Core.add(token_embedding.call(x), pos)

      kv_cache ||= Array.new(blocks.length)
      cross_qk = Array.new(blocks.length)
      blocks.each_with_index do |block, idx|
        h, kv_cache[idx], cross_qk[idx] = block.call(h, xa: xa, mask: mask, kv_cache: kv_cache[idx])
      end

      h = ln.call(h)
      [output_proj.call(h), kv_cache, cross_qk]
    end

    private

    def validate_token_ids!(x)
      token_count = x.shape.reduce(1, :*)
      return if token_count.zero?

      min_id = MLX::Core.min(x).item.to_i
      max_id = MLX::Core.max(x).item.to_i
      vocab_size = token_embedding.weight.shape[0]

      return if min_id >= 0 && max_id < vocab_size

      raise ArgumentError, "token ids out of range for decoder vocab #{vocab_size}: min=#{min_id}, max=#{max_id}"
    end
  end

  class Whisper < MLX::NN::Module
    attr_reader :dims

    def initialize(dims, dtype: MLX::Core.float16)
      super()
      @dims = dims.is_a?(ModelDimensions) ? dims : ModelDimensions.from_hash(dims)
      self.encoder = AudioEncoder.new(
        @dims.n_mels,
        @dims.n_audio_ctx,
        @dims.n_audio_state,
        @dims.n_audio_head,
        @dims.n_audio_layer,
        dtype: dtype
      )
      self.decoder = TextDecoder.new(
        @dims.n_vocab,
        @dims.n_text_ctx,
        @dims.n_text_state,
        @dims.n_text_head,
        @dims.n_text_layer,
        dtype: dtype
      )

      all_heads = Array.new(@dims.n_text_layer) { Array.new(@dims.n_text_head, false) }
      (@dims.n_text_layer / 2...@dims.n_text_layer).each do |l|
        @dims.n_text_head.times { |h| all_heads[l][h] = true }
      end
      pairs = []
      all_heads.each_with_index do |row, l|
        row.each_with_index { |flag, h| pairs << [l, h] if flag }
      end
      self.alignment_heads = MLX::Core.array(pairs, MLX::Core.int32)
    end

    def set_alignment_heads(dump)
      if dump.respond_to?(:shape)
        self.alignment_heads = dump
      elsif dump.is_a?(String) || dump.is_a?(Array)
        self.alignment_heads = MLX::Core.array(dump, MLX::Core.int32)
      else
        raise ArgumentError, "Unsupported alignment head type: #{dump.class}"
      end
    end

    def embed_audio(mel)
      encoder.call(mel)
    end

    def logits(tokens, audio_features)
      decoder.call(tokens, audio_features)[0]
    end

    def forward_with_cross_qk(mel, tokens)
      logits, _cache, cross_qk = decoder.call(tokens, encoder.call(mel))
      [logits, cross_qk]
    end

    def call(mel, tokens)
      decoder.call(tokens, encoder.call(mel))[0]
    end

    def is_multilingual
      dims.n_vocab >= 51_865
    end

    def num_languages
      dims.n_vocab - 51_765 - (is_multilingual ? 1 : 0)
    end

    def detect_language(mel, tokenizer: nil)
      Decoding.detect_language(self, mel, tokenizer: tokenizer)
    end

    def decode(mel, options = nil, **kwargs)
      Decoding.decode(self, mel, options, **kwargs)
    end
  end
end
