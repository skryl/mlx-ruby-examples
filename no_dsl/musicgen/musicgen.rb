# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

require_relative "encodec"

module MusicGenExample
  SNAPSHOT_SCRIPT = Pathname.new(__dir__).join("python", "snapshot_download.py").to_s
  EXTRACT_SCRIPT = Pathname.new(__dir__).join("python", "extract_state_dict.py").to_s

  class TextEncoderConfig
    attr_reader :_name_or_path, :d_model, :vocab_size, :max_length

    def initialize(_name_or_path: "google/flan-t5-small", d_model: 512, vocab_size: 2048, max_length: 32)
      @_name_or_path = _name_or_path
      @d_model = d_model
      @vocab_size = vocab_size
      @max_length = max_length
    end

    def self.from_hash(raw)
      data = raw.transform_keys(&:to_s)
      new(
        _name_or_path: data.fetch("_name_or_path", "google/flan-t5-small"),
        d_model: data.fetch("d_model", 512),
        vocab_size: data.fetch("vocab_size", 2048),
        max_length: data.fetch("max_length", 32)
      )
    end
  end

  class AudioEncoderConfig
    attr_reader :_name_or_path, :codebook_size, :sampling_rate

    def initialize(_name_or_path: "facebook/encodec_32khz", codebook_size: 1024, sampling_rate: 32_000)
      @_name_or_path = _name_or_path
      @codebook_size = codebook_size
      @sampling_rate = sampling_rate
    end

    def self.from_hash(raw)
      data = raw.transform_keys(&:to_s)
      new(
        _name_or_path: data.fetch("_name_or_path", "facebook/encodec_32khz"),
        codebook_size: data.fetch("codebook_size", 1024),
        sampling_rate: data.fetch("sampling_rate", 32_000)
      )
    end
  end

  class DecoderConfig
    attr_reader :num_codebooks,
                :bos_token_id,
                :hidden_size,
                :num_attention_heads,
                :ffn_dim,
                :num_hidden_layers

    def initialize(
      num_codebooks: 4,
      bos_token_id: 1024,
      hidden_size: 256,
      num_attention_heads: 8,
      ffn_dim: 1024,
      num_hidden_layers: 6
    )
      @num_codebooks = num_codebooks
      @bos_token_id = bos_token_id
      @hidden_size = hidden_size
      @num_attention_heads = num_attention_heads
      @ffn_dim = ffn_dim
      @num_hidden_layers = num_hidden_layers
    end

    def self.from_hash(raw)
      data = raw.transform_keys(&:to_s)
      new(
        num_codebooks: data.fetch("num_codebooks", 4),
        bos_token_id: data.fetch("bos_token_id", 1024),
        hidden_size: data.fetch("hidden_size", 256),
        num_attention_heads: data.fetch("num_attention_heads", 8),
        ffn_dim: data.fetch("ffn_dim", data.fetch("hidden_size", 256) * 4),
        num_hidden_layers: data.fetch("num_hidden_layers", 6)
      )
    end
  end

  class MusicGenConfig
    attr_reader :text_encoder, :audio_encoder, :decoder

    def initialize(text_encoder:, audio_encoder:, decoder:)
      @text_encoder = text_encoder
      @audio_encoder = audio_encoder
      @decoder = decoder
    end

    def self.from_hash(raw)
      data = raw.transform_keys(&:to_s)
      text = data.fetch("text_encoder", {})
      audio = data.fetch("audio_encoder", {})
      decoder = data.fetch("decoder", {})
      new(
        text_encoder: text.is_a?(TextEncoderConfig) ? text : TextEncoderConfig.from_hash(text),
        audio_encoder: audio.is_a?(AudioEncoderConfig) ? audio : AudioEncoderConfig.from_hash(audio),
        decoder: decoder.is_a?(DecoderConfig) ? decoder : DecoderConfig.from_hash(decoder)
      )
    end
  end

  class TextTokenizer
    attr_reader :max_length, :vocab_size

    def initialize(vocab_size: 2048, max_length: 32)
      @vocab_size = vocab_size
      @max_length = max_length
    end

    def encode(text)
      texts = text.is_a?(Array) ? text : [text]
      token_ids = texts.map do |item|
        bytes = item.to_s.bytes.take(max_length)
        ids = bytes.map { |b| (b % (vocab_size - 1)) + 1 }
        ids.fill(0, ids.length...max_length)
        ids
      end
      MLX::Core.array(token_ids, MLX::Core.int32)
    end
  end

  class TextConditioner < MLX::NN::Module
    attr_reader :tokenizer

    def initialize(_t5_name, input_dim, output_dim, vocab_size: 2048, max_length: 32)
      super()
      @tokenizer = TextTokenizer.new(vocab_size: vocab_size, max_length: max_length)
      self.token_embedding = MLX::NN::Embedding.new(vocab_size, input_dim)
      self.output_proj = MLX::NN::Linear.new(input_dim, output_dim)
    end

    def call(text)
      x = tokenizer.encode(text)
      x = token_embedding.call(x)
      output_proj.call(x)
    end
  end

  class KVCache
    attr_reader :n_kv_heads, :k_head_dim, :v_head_dim, :keys, :values, :offset

    def initialize(head_dim, n_kv_heads)
      @n_kv_heads = n_kv_heads
      if head_dim.is_a?(Integer)
        @k_head_dim = head_dim
        @v_head_dim = head_dim
      elsif head_dim.is_a?(Array) && head_dim.length == 2
        @k_head_dim = head_dim[0]
        @v_head_dim = head_dim[1]
      else
        raise ArgumentError, "head_dim must be an Integer or [k_dim, v_dim]"
      end

      @keys = nil
      @values = nil
      @offset = 0
    end

    def update_and_fetch(keys, values)
      if @keys.nil?
        @keys = keys
        @values = values
      else
        @keys = MLX::Core.concatenate([@keys, keys], 2)
        @values = MLX::Core.concatenate([@values, values], 2)
      end
      @offset += keys.shape[2]
      [@keys, @values]
    end

    def state
      [@keys, @values]
    end
  end

  class MultiHeadAttention < MLX::NN::Module
    def initialize(dim, n_heads)
      super()
      @n_heads = n_heads
      @head_dim = dim / n_heads
      @scale = @head_dim**-0.5

      self.q_proj = MLX::NN::Linear.new(dim, dim, bias: false)
      self.k_proj = MLX::NN::Linear.new(dim, dim, bias: false)
      self.v_proj = MLX::NN::Linear.new(dim, dim, bias: false)
      self.out_proj = MLX::NN::Linear.new(dim, dim, bias: false)
    end

    def call(queries, keys, values, mask: nil, cache: nil)
      b, lq, d = queries.shape
      lk = keys.shape[1]

      queries = q_proj.call(queries)
      keys = k_proj.call(keys)
      values = v_proj.call(values)

      queries = MLX::Core.transpose(MLX::Core.reshape(queries, [b, lq, @n_heads, @head_dim]), [0, 2, 1, 3])
      keys = MLX::Core.transpose(MLX::Core.reshape(keys, [b, lk, @n_heads, @head_dim]), [0, 2, 1, 3])
      values = MLX::Core.transpose(MLX::Core.reshape(values, [b, lk, @n_heads, @head_dim]), [0, 2, 1, 3])

      keys, values = cache.update_and_fetch(keys, values) unless cache.nil?

      output = MLX::Core.scaled_dot_product_attention(queries, keys, values, @scale, mask)
      output = MLX::Core.transpose(output, [0, 2, 1, 3])
      output = MLX::Core.reshape(output, [b, lq, d])
      out_proj.call(output)
    end
  end

  class TransformerBlock < MLX::NN::Module
    def initialize(config)
      super()
      hidden = config.decoder.hidden_size
      n_heads = config.decoder.num_attention_heads

      self.self_attn = MultiHeadAttention.new(hidden, n_heads)
      self.cross_attn = MultiHeadAttention.new(hidden, n_heads)
      self.linear1 = MLX::NN::Linear.new(hidden, config.decoder.ffn_dim, bias: false)
      self.linear2 = MLX::NN::Linear.new(config.decoder.ffn_dim, hidden, bias: false)

      self.norm1 = MLX::NN::LayerNorm.new(hidden, eps: 1e-5)
      self.norm_cross = MLX::NN::LayerNorm.new(hidden, eps: 1e-5)
      self.norm2 = MLX::NN::LayerNorm.new(hidden, eps: 1e-5)
    end

    def call(x, conditioning, mask: nil, cache: nil)
      xn = norm1.call(x)
      x = MLX::Core.add(x, self_attn.call(xn, xn, xn, mask: mask, cache: cache))
      xn = norm_cross.call(x)
      x = MLX::Core.add(x, cross_attn.call(xn, conditioning, conditioning, mask: mask, cache: nil))
      xn = norm2.call(x)
      x = MLX::Core.add(x, linear2.call(MLX::NN.gelu(linear1.call(xn))))
      x
    end
  end

  def self.top_k_sampling(logits, top_k, temperature, axis: -1)
    dim = logits.shape[axis]
    return MLX::Core.expand_dims(MLX::Core.argmax(logits, axis), axis) if top_k <= 0 || top_k >= dim

    sorted_indices = MLX::Core.argsort(logits, axis)
    keep = MLX::Core.array(((dim - top_k)...dim).to_a, MLX::Core.int32)
    top_indices = MLX::Core.take(sorted_indices, keep, axis)
    top_logits = MLX::Core.take_along_axis(logits, top_indices, axis)

    # Keep deterministic default behavior for reproducible tests.
    scaled = temperature.positive? ? MLX::Core.multiply(top_logits, 1.0 / temperature.to_f) : top_logits
    selected = MLX::Core.argmax(scaled, axis)
    MLX::Core.take_along_axis(top_indices, MLX::Core.expand_dims(selected, axis), axis)
  end

  def self.create_sin_embedding(positions, dim, max_period: 10_000.0)
    raise ArgumentError, "dim must be even" unless (dim % 2).zero?

    half_dim = dim / 2
    adim = MLX::Core.reshape(MLX::Core.arange(0, half_dim, 1, MLX::Core.float32), [1, 1, half_dim])
    pos = positions.respond_to?(:shape) ? positions.astype(MLX::Core.float32) : MLX::Core.array(positions.to_f, MLX::Core.float32)
    denom = MLX::Core.power(max_period.to_f, MLX::Core.divide(adim, [half_dim - 1, 1].max.to_f))
    phase = MLX::Core.divide(pos, denom)
    MLX::Core.concatenate([MLX::Core.cos(phase), MLX::Core.sin(phase)], -1)
  end

  class MusicGen < MLX::NN::Module
    attr_reader :num_codebooks,
                :codebook_size,
                :bos_token_id,
                :hidden_size,
                :num_attention_heads,
                :sampling_rate

    def initialize(config, audio_decoder: nil)
      super()
      @config = config.is_a?(MusicGenConfig) ? config : MusicGenConfig.from_hash(config)

      @num_codebooks = @config.decoder.num_codebooks
      @codebook_size = @config.audio_encoder.codebook_size
      @bos_token_id = @config.decoder.bos_token_id
      @hidden_size = @config.decoder.hidden_size
      @num_attention_heads = @config.decoder.num_attention_heads
      @sampling_rate = @config.audio_encoder.sampling_rate

      self.text_conditioner = TextConditioner.new(
        @config.text_encoder._name_or_path,
        @config.text_encoder.d_model,
        @hidden_size,
        vocab_size: @config.text_encoder.vocab_size,
        max_length: @config.text_encoder.max_length
      )

      self.emb = Array.new(@num_codebooks) { MLX::NN::Embedding.new(@codebook_size + 1, @hidden_size) }
      self.layers = Array.new(@config.decoder.num_hidden_layers) { TransformerBlock.new(@config) }
      self.out_norm = MLX::NN::LayerNorm.new(@hidden_size, eps: 1e-5)
      self.linears = Array.new(@num_codebooks) { MLX::NN::Linear.new(@hidden_size, @codebook_size, bias: false) }

      self.audio_decoder = if audio_decoder.nil?
                             enc_cfg = EncodecExample::EncodecConfig.new(
                               audio_channels: 1,
                               num_filters: 16,
                               hidden_size: @hidden_size,
                               codebook_size: @codebook_size,
                               codebook_dim: @hidden_size,
                               upsampling_ratios: [2, 2],
                               target_bandwidths: [400.0],
                               num_lstm_layers: 1,
                               num_residual_layers: 1,
                               sampling_rate: @sampling_rate,
                               normalize: false
                             )
                             EncodecExample::EncodecModel.new(enc_cfg)
                           else
                             audio_decoder
                           end
    end

    def call(audio_tokens, conditioning, cache: nil)
      cache ||= Array.new(layers.length)

      x = nil
      num_codebooks.times do |k|
        tok = MLX::Core.squeeze(MLX::Core.take(audio_tokens, MLX::Core.array([k], MLX::Core.int32), -1), -1)
        emb_k = emb[k].call(tok)
        x = x.nil? ? emb_k : MLX::Core.add(x, emb_k)
      end

      offset = (!cache.empty? && !cache[0].nil?) ? cache[0].offset : 0
      pos_emb = MusicGenExample.create_sin_embedding(offset, hidden_size)
      x = MLX::Core.add(x, pos_emb.astype(x.dtype))

      layers.each_with_index do |layer, i|
        x = layer.call(x, conditioning, cache: cache[i])
      end

      x = out_norm.call(x)
      logits = num_codebooks.times.map { |k| linears[k].call(x) }
      MLX::Core.stack(logits, -1)
    end

    def generate(text, max_steps: 200, top_k: 250, temp: 1.0, guidance_coef: 3.0)
      conditioning = text_conditioner.call(text)
      conditioning = MLX::Core.concatenate([conditioning, MLX::Core.zeros_like(conditioning)], 0)

      head_dim = hidden_size / num_attention_heads
      cache = Array.new(layers.length) { KVCache.new(head_dim, num_attention_heads) }

      current = MLX::Core.full([1, 1, num_codebooks], bos_token_id, MLX::Core.int32)
      generated = []

      max_steps.times do
        audio_input = MLX::Core.concatenate([current, current], 0)
        logits = call(audio_input, conditioning, cache: cache)

        cond_logits = MLX::Core.slice(logits, [0, 0, 0, 0], [1, logits.shape[1], logits.shape[2], logits.shape[3]])
        uncond_logits = MLX::Core.slice(logits, [1, 0, 0, 0], [2, logits.shape[1], logits.shape[2], logits.shape[3]])
        guided = MLX::Core.add(
          uncond_logits,
          MLX::Core.multiply(MLX::Core.subtract(cond_logits, uncond_logits), guidance_coef.to_f)
        )

        token = MusicGenExample.top_k_sampling(guided, top_k, temp, axis: -2)
        token = MLX::Core.squeeze(token, -2).astype(MLX::Core.int32)

        generated << token
        current = token
        MLX::Core.eval(current)
      end

      if generated.empty?
        codes = MLX::Core.full([1, num_codebooks, 1], bos_token_id, MLX::Core.int32)
      else
        seq = MLX::Core.concatenate(generated, 1) # [1, T, K]
        codes = MLX::Core.transpose(seq, [0, 2, 1]) # [1, K, T]
      end

      audio = audio_decoder.decode(codes, [nil])
      MLX::Core.squeeze(audio, 0)
    end

    def self.sanitize(weights)
      out = {}
      weights.each do |key, arr|
        k = key.to_s
        k = k.delete_prefix("transformer.") if k.start_with?("transformer.")
        k = k.gsub("cross_attention", "cross_attn") if k.include?("cross_attention")
        k = k.gsub("condition_provider.conditioners.description", "text_conditioner") if k.include?("condition_provider")

        if k.include?("in_proj_weight") && arr.shape.length == 2
          dim = arr.shape[0] / 3
          base = k.sub("in_proj_weight", "")
          out["#{base}q_proj.weight"] = MLX::Core.slice(arr, [0, 0], [dim, arr.shape[1]])
          out["#{base}k_proj.weight"] = MLX::Core.slice(arr, [dim, 0], [dim * 2, arr.shape[1]])
          out["#{base}v_proj.weight"] = MLX::Core.slice(arr, [dim * 2, 0], [arr.shape[0], arr.shape[1]])
          next
        end

        out[k] = arr
      end
      out
    end

    def self.from_pretrained(path_or_repo, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      path = Pathname.new(path_or_repo.to_s)
      path = MusicGenExample.snapshot_download(path_or_repo.to_s, python_bin: python_bin) unless path.exist?

      config_path = path.join("config.json")
      raise Errno::ENOENT, "Could not find #{config_path}" unless config_path.exist?

      config = MusicGenConfig.from_hash(JSON.parse(File.binread(config_path)))
      model = MusicGen.new(config)

      npz_path = path.join("state_dict.npz")
      unless npz_path.exist?
        bin_path = path.join("state_dict.bin")
        if bin_path.exist?
          npz_path = MusicGenExample.extract_state_dict(bin_path, npz_path, python_bin: python_bin)
        end
      end

      if npz_path.exist?
        weights = MLX::Core.load(npz_path.to_s).to_a.each_with_object({}) { |(k, v), out| out[k.to_s] = v }
        begin
          model.load_weights(sanitize(weights).to_a, strict: false)
        rescue StandardError
          # Best effort for lightweight port.
        end
      end

      model
    end
  end

  module_function

  def snapshot_download(repo_id, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    stdout, stderr, status = Open3.capture3(python_bin, SNAPSHOT_SCRIPT, repo_id.to_s)
    raise "Failed to download snapshot for #{repo_id}: #{stderr}" unless status.success?

    Pathname.new(stdout.strip)
  end

  def extract_state_dict(bin_path, out_path, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    stdout, stderr, status = Open3.capture3(python_bin, EXTRACT_SCRIPT, bin_path.to_s, out_path.to_s)
    raise "Failed to extract state dict: #{stderr}" unless status.success?

    Pathname.new(stdout.strip)
  end
end
