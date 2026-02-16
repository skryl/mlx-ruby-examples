# frozen_string_literal: true

require "json"
require "open3"
require "pathname"

# Ensure local DSL extensions are on the load path when running from repository root.

require "mlx"
require "mlx/dsl"

module EncodecExample
  SNAPSHOT_SCRIPT = Pathname.new(__dir__).join("python", "snapshot_download.py").to_s

  module Ops
    module_function

    def sigmoid(x)
      MLX::Core.divide(1.0, MLX::Core.add(1.0, MLX::Core.exp(MLX::Core.multiply(-1.0, x))))
    end

    def slice_time_3d(x, offset, length)
      MLX::Core.slice(x, [0, offset, 0], [x.shape[0], offset + length, x.shape[2]])
    end

    def slice_time_2d(x, offset, length)
      MLX::Core.slice(x, [0, offset], [x.shape[0], offset + length])
    end

    def strided_take(x, stride)
      return x if stride <= 1

      idx = MLX::Core.arange(0, x.shape[1], stride, MLX::Core.int32)
      return x if idx.shape[0].zero?

      MLX::Core.take(x, idx, 1)
    end

    def repeat_along_time(x, repeats)
      return x if repeats <= 1

      b, t, c = x.shape
      y = MLX::Core.expand_dims(x, 2)
      y = MLX::Core.concatenate(Array.new(repeats, y), 2)
      MLX::Core.reshape(y, [b, t * repeats, c])
    end

    def as_array(x, dtype: MLX::Core.float32)
      return x if x.respond_to?(:shape)

      MLX::Core.array(x, dtype)
    end
  end

  class EncodecConfig
    include MLX::DSL::ConfigSchema

    field :use_causal_conv, [TrueClass, FalseClass], default: true
    field :pad_mode, String, default: "reflect"
    field :norm_type, String, default: "weight_norm"
    field :trim_right_ratio, [Integer, Float], default: 1.0
    field :num_lstm_layers, Integer, default: 2
    field :residual_kernel_size, Integer, default: 3
    field :compress, Integer, default: 2
    field :use_conv_shortcut, [TrueClass, FalseClass], default: true
    field :audio_channels, Integer, default: 2
    field :num_filters, Integer, default: 32
    field :kernel_size, Integer, default: 7
    field :upsampling_ratios, Array, default: [8, 5, 4, 2]
    field :num_residual_layers, Integer, default: 1
    field :dilation_growth_rate, Integer, default: 2
    field :hidden_size, Integer, default: 128
    field :last_kernel_size, Integer, default: 7
    field :codebook_size, Integer, default: 1024
    field :codebook_dim, Integer, default: ->(cfg) { cfg.hidden_size }
    field :sampling_rate, Integer, default: 48_000
    field :target_bandwidths, Array, default: [1.5, 3.0, 6.0, 12.0]
    field :chunk_length_s, [Integer, Float, NilClass], default: nil
    field :overlap, [Integer, Float, NilClass], default: nil
    field :normalize, [TrueClass, FalseClass], default: true

    def self.from_hash(raw)
      super
    end
  end

  class LSTM < MLX::NN::Module
    attr_reader :hidden_size

    def initialize(input_size, hidden_size, bias: true)
      super()
      @hidden_size = hidden_size
      self.wx = MLX::NN::Linear.new(input_size, hidden_size * 4, bias: bias)
      self.wh = MLX::NN::Linear.new(hidden_size, hidden_size * 4, bias: false)
    end

    def call(x, hidden: nil, cell: nil)
      b = x.shape[0]
      t = x.shape[1]
      dtype = x.dtype

      hidden = MLX::Core.zeros([b, hidden_size], dtype) if hidden.nil?
      cell = MLX::Core.zeros([b, hidden_size], dtype) if cell.nil?

      outputs = []
      t.times do |step|
        xt = MLX::Core.squeeze(Ops.slice_time_3d(x, step, 1), 1)
        gates = MLX::Core.add(wx.call(xt), wh.call(hidden))
        i, f, g, o = MLX::Core.split(gates, [hidden_size, hidden_size * 2, hidden_size * 3], -1)
        i = Ops.sigmoid(i)
        f = Ops.sigmoid(f)
        g = MLX::Core.tanh(g)
        o = Ops.sigmoid(o)
        cell = MLX::Core.add(MLX::Core.multiply(f, cell), MLX::Core.multiply(i, g))
        hidden = MLX::Core.multiply(o, MLX::Core.tanh(cell))
        outputs << hidden
      end

      MLX::Core.stack(outputs, 1)
    end
  end

  class EncodecConv1d < MLX::NN::Module
    def initialize(config, in_channels, out_channels, _kernel_size, stride: 1, dilation: 1)
      super()
      _ = dilation
      @stride = stride
      @norm_type = config.norm_type
      self.proj = MLX::NN::Linear.new(in_channels, out_channels)
      self.norm = MLX::NN::LayerNorm.new(out_channels) if @norm_type == "time_group_norm"
    end

    def call(hidden_states)
      x = Ops.strided_take(hidden_states, @stride)
      x = proj.call(x)
      x = norm.call(x) if @norm_type == "time_group_norm"
      x
    end
  end

  class EncodecConvTranspose1d < MLX::NN::Module
    def initialize(config, in_channels, out_channels, kernel_size, stride: 1)
      super()
      @causal = config.use_causal_conv
      @trim_right_ratio = config.trim_right_ratio
      @stride = stride
      @padding_total = [kernel_size - stride, 0].max
      @norm_type = config.norm_type
      self.proj = MLX::NN::Linear.new(in_channels, out_channels)
      self.norm = MLX::NN::LayerNorm.new(out_channels) if @norm_type == "time_group_norm"
    end

    def call(hidden_states)
      x = Ops.repeat_along_time(hidden_states, @stride)
      x = proj.call(x)
      x = norm.call(x) if @norm_type == "time_group_norm"

      return x if @padding_total <= 0

      padding_right = if @causal
                        (@padding_total * @trim_right_ratio).ceil
                      else
                        @padding_total / 2
                      end
      padding_left = @padding_total - padding_right

      end_idx = [x.shape[1] - padding_right, padding_left + 1].max
      MLX::Core.slice(x, [0, padding_left, 0], [x.shape[0], end_idx, x.shape[2]])
    end
  end

  class EncodecLSTM < MLX::NN::Module
    def initialize(config, dimension)
      super()
      self.layers = Array.new(config.num_lstm_layers) { LSTM.new(dimension, dimension) }
    end

    def call(hidden_states)
      x = hidden_states
      x = MLX::DSL.run_stack(layers, x)
      MLX::Core.add(x, hidden_states)
    end
  end

  class EncodecResnetBlock < MLX::NN::Module
    def initialize(config, dim, _dilations)
      super()
      hidden = [dim / config.compress, 1].max

      self.act1 = MLX::NN::GELU.new
      self.conv1 = EncodecConv1d.new(config, dim, hidden, config.residual_kernel_size)
      self.act2 = MLX::NN::GELU.new
      self.conv2 = EncodecConv1d.new(config, hidden, dim, 1)

      if config.use_conv_shortcut
        self.shortcut = EncodecConv1d.new(config, dim, dim, 1)
      else
        self.shortcut = nil
      end
    end

    def call(hidden_states)
      residual = shortcut.nil? ? hidden_states : shortcut.call(hidden_states)
      h = act1.call(hidden_states)
      h = conv1.call(h)
      h = act2.call(h)
      h = conv2.call(h)
      MLX::Core.add(residual, h)
    end
  end

  class EncodecEncoder < MLX::NN::Module
    def initialize(config)
      super()
      model = [EncodecConv1d.new(config, config.audio_channels, config.num_filters, config.kernel_size)]
      scaling = 1

      config.upsampling_ratios.reverse.each do |ratio|
        current_scale = scaling * config.num_filters
        config.num_residual_layers.times do |j|
          model << EncodecResnetBlock.new(config, current_scale, [config.dilation_growth_rate**j, 1])
        end
        model << MLX::NN::GELU.new
        model << EncodecConv1d.new(config, current_scale, current_scale * 2, ratio * 2, stride: ratio)
        scaling *= 2
      end

      model << EncodecLSTM.new(config, scaling * config.num_filters)
      model << MLX::NN::GELU.new
      model << EncodecConv1d.new(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)
      self.layers = model
    end

    def call(hidden_states)
      MLX::DSL.run_stack(layers, hidden_states)
    end
  end

  class EncodecDecoder < MLX::NN::Module
    def initialize(config)
      super()
      scaling = 2**config.upsampling_ratios.length
      model = [
        EncodecConv1d.new(config, config.hidden_size, scaling * config.num_filters, config.kernel_size),
        EncodecLSTM.new(config, scaling * config.num_filters)
      ]

      config.upsampling_ratios.each do |ratio|
        current_scale = scaling * config.num_filters
        model << MLX::NN::GELU.new
        model << EncodecConvTranspose1d.new(
          config,
          current_scale,
          current_scale / 2,
          ratio * 2,
          stride: ratio
        )
        config.num_residual_layers.times do |j|
          model << EncodecResnetBlock.new(config, current_scale / 2, [config.dilation_growth_rate**j, 1])
        end
        scaling /= 2
      end

      model << MLX::NN::GELU.new
      model << EncodecConv1d.new(config, config.num_filters, config.audio_channels, config.last_kernel_size)
      self.layers = model
    end

    def call(hidden_states)
      MLX::DSL.run_stack(layers, hidden_states)
    end
  end

  class EncodecEuclideanCodebook < MLX::NN::Module
    def initialize(config)
      super()
      self.embed = MLX::Core.zeros([config.codebook_size, config.codebook_dim], MLX::Core.float32)
    end

    def quantize(hidden_states)
      embed_t = MLX::Core.transpose(embed, [1, 0])
      state_norm = MLX::Core.expand_dims(MLX::Core.sum(MLX::Core.square(hidden_states), -1), -1)
      dot = MLX::Core.matmul(hidden_states, embed_t)
      embed_norm = MLX::Core.expand_dims(MLX::Core.sum(MLX::Core.square(embed), -1), 0)
      dist = MLX::Core.multiply(-1.0, MLX::Core.add(MLX::Core.subtract(state_norm, MLX::Core.multiply(2.0, dot)), embed_norm))
      MLX::Core.argmax(dist, -1)
    end

    def encode(hidden_states)
      shape = hidden_states.shape
      rows = shape[0...-1].reduce(1, :*)
      flat = MLX::Core.reshape(hidden_states, [rows, shape[-1]])
      indices = quantize(flat)
      MLX::Core.reshape(indices, shape[0...-1])
    end

    def decode(embed_ind)
      rows = embed_ind.shape.reduce(1, :*)
      flat = MLX::Core.reshape(embed_ind, [rows])
      gathered = MLX::Core.take(embed, flat, 0)
      MLX::Core.reshape(gathered, embed_ind.shape + [embed.shape[1]])
    end
  end

  class EncodecVectorQuantization < MLX::NN::Module
    def initialize(config)
      super()
      self.codebook = EncodecEuclideanCodebook.new(config)
    end

    def encode(hidden_states)
      codebook.encode(hidden_states)
    end

    def decode(embed_ind)
      codebook.decode(embed_ind)
    end
  end

  class EncodecResidualVectorQuantizer < MLX::NN::Module
    attr_reader :codebook_size, :frame_rate, :num_quantizers

    def initialize(config)
      super()
      @codebook_size = config.codebook_size
      hop_length = config.upsampling_ratios.reduce(1, :*)
      @frame_rate = (config.sampling_rate.to_f / hop_length).ceil
      @num_quantizers = ((1000 * config.target_bandwidths[-1]) / (frame_rate * 10.0)).floor
      @num_quantizers = [@num_quantizers, 1].max
      self.layers = Array.new(@num_quantizers) { EncodecVectorQuantization.new(config) }
    end

    def get_num_quantizers_for_bandwidth(bandwidth = nil)
      bw_per_q = Math.log2(codebook_size) * frame_rate
      out = num_quantizers
      if !bandwidth.nil? && bandwidth.positive?
        out = [(bandwidth * 1000 / bw_per_q).floor, 1].max
      end
      [out, num_quantizers].min
    end

    def encode(embeddings, bandwidth = nil)
      num_q = get_num_quantizers_for_bandwidth(bandwidth)
      residual = embeddings
      all_indices = []

      layers.first(num_q).each do |layer|
        indices = layer.encode(residual)
        quantized = layer.decode(indices)
        residual = MLX::Core.subtract(residual, quantized)
        all_indices << indices
      end

      MLX::Core.stack(all_indices, 1)
    end

    def decode(codes)
      quantized_out = nil
      codes.shape[1].times do |i|
        idx = MLX::Core.array([i], MLX::Core.int32)
        indices = MLX::Core.squeeze(MLX::Core.take(codes, idx, 1), 1)
        quantized = layers[i].decode(indices)
        quantized_out = quantized_out.nil? ? quantized : MLX::Core.add(quantized_out, quantized)
      end
      quantized_out
    end
  end

  class EncodecModel < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config.is_a?(EncodecConfig) ? config : EncodecConfig.from_hash(config)
      self.encoder = EncodecEncoder.new(@config)
      self.decoder = EncodecDecoder.new(@config)
      self.quantizer = EncodecResidualVectorQuantizer.new(@config)
    end

    def _encode_frame(input_values, bandwidth, padding_mask)
      length = input_values.shape[1]
      duration = length.to_f / config.sampling_rate

      if !config.chunk_length_s.nil? && duration > (1e-5 + config.chunk_length_s)
        raise "Duration of frame (#{duration}) is longer than chunk #{config.chunk_length_s}"
      end

      scale = nil
      x = input_values
      if config.normalize
        x = MLX::Core.multiply(x, MLX::Core.expand_dims(padding_mask, -1))
        mono = MLX::Core.divide(MLX::Core.expand_dims(MLX::Core.sum(x, -1), -1), x.shape[2].to_f)
        scale = MLX::Core.add(
          MLX::Core.sqrt(MLX::Core.expand_dims(MLX::Core.mean(MLX::Core.square(mono), 1), 1)),
          1e-8
        )
        x = MLX::Core.divide(x, scale)
      end

      embeddings = encoder.call(x)
      codes = quantizer.encode(embeddings, bandwidth)
      [codes, scale]
    end

    def encode(input_values, padding_mask = nil, bandwidth: nil)
      bandwidth = config.target_bandwidths[0] if bandwidth.nil?
      unless config.target_bandwidths.include?(bandwidth)
        raise ArgumentError,
              "This model does not support bandwidth #{bandwidth}. Select one of #{config.target_bandwidths.inspect}."
      end

      input_length = input_values.shape[1]
      channels = input_values.shape[2]
      unless channels.between?(1, 2)
        raise ArgumentError, "Number of audio channels must be 1 or 2, but got #{channels}"
      end

      chunk_len = chunk_length || input_length
      stride = chunk_stride || input_length

      padding_mask = MLX::Core.ones(input_values.shape[0...2], MLX::Core.bool_) if padding_mask.nil?

      step = chunk_len - stride
      if (input_length % stride) != step
        raise ArgumentError,
              "Input length is not properly padded for batched chunked encoding. Pad inputs before calling encode."
      end

      encoded_frames = []
      scales = []
      offset = 0
      while offset < (input_length - step)
        mask = Ops.slice_time_2d(padding_mask, offset, chunk_len)
        frame = Ops.slice_time_3d(input_values, offset, chunk_len)
        encoded_frame, scale = _encode_frame(frame, bandwidth, mask)
        encoded_frames << encoded_frame
        scales << scale
        offset += stride
      end

      [MLX::Core.stack(encoded_frames, 0), scales]
    end

    def self.linear_overlap_add(frames, stride)
      raise ArgumentError, "frames cannot be empty" if frames.empty?

      n = frames[0].shape[0]
      c = frames[0].shape[2]
      first_frame_len = frames[0].shape[1]
      total_size = stride * (frames.length - 1) + frames[-1].shape[1]

      weight = Array.new(first_frame_len) do |i|
        t = (i + 1).to_f / (first_frame_len + 1)
        0.5 - (t - 0.5).abs
      end

      out = Array.new(n) { Array.new(total_size) { Array.new(c, 0.0) } }
      sum_weight = Array.new(total_size, 0.0)

      offset = 0
      frames.each do |frame|
        frame_array = frame.to_a
        frame_len = frame.shape[1]

        n.times do |bi|
          frame_len.times do |ti|
            w = weight[ti]
            c.times do |ci|
              out[bi][offset + ti][ci] += frame_array[bi][ti][ci] * w
            end
          end
        end

        frame_len.times do |ti|
          sum_weight[offset + ti] += weight[ti]
        end

        offset += stride
      end

      n.times do |bi|
        total_size.times do |ti|
          denom = sum_weight[ti].zero? ? 1.0 : sum_weight[ti]
          c.times do |ci|
            out[bi][ti][ci] /= denom
          end
        end
      end

      MLX::Core.array(out, frames[0].dtype)
    end

    def _decode_frame(codes, scale = nil)
      embeddings = quantizer.decode(codes)
      outputs = decoder.call(embeddings)
      outputs = MLX::Core.multiply(outputs, scale) unless scale.nil?
      outputs
    end

    def channels
      config.audio_channels
    end

    def sampling_rate
      config.sampling_rate
    end

    def chunk_length
      return nil if config.chunk_length_s.nil?

      (config.chunk_length_s * config.sampling_rate).to_i
    end

    def chunk_stride
      return nil if config.chunk_length_s.nil? || config.overlap.nil?

      [1, ((1.0 - config.overlap) * chunk_length).to_i].max
    end

    def decode(audio_codes, audio_scales, padding_mask = nil)
      chunk_len = chunk_length

      if chunk_len.nil?
        if audio_codes.shape.length == 4
          if audio_codes.shape[1] != 1
            raise ArgumentError, "Expected batch size 1 for non-chunked decode, got #{audio_codes.shape[1]}"
          end
          frame_codes = MLX::Core.squeeze(MLX::Core.take(audio_codes, MLX::Core.array([0], MLX::Core.int32), 0), 0)
        else
          frame_codes = audio_codes
        end

        scale = audio_scales.is_a?(Array) ? audio_scales[0] : audio_scales
        audio_values = _decode_frame(frame_codes, scale)
      else
        decoded_frames = []
        audio_codes.shape[0].times do |i|
          frame = MLX::Core.squeeze(MLX::Core.take(audio_codes, MLX::Core.array([i], MLX::Core.int32), 0), 0)
          scale = audio_scales[i]
          decoded_frames << _decode_frame(frame, scale)
        end

        audio_values = self.class.linear_overlap_add(decoded_frames, chunk_stride || 1)
      end

      if !padding_mask.nil? && padding_mask.shape[1] < audio_values.shape[1]
        audio_values = MLX::Core.slice(
          audio_values,
          [0, 0, 0],
          [audio_values.shape[0], padding_mask.shape[1], audio_values.shape[2]]
        )
      end

      audio_values
    end

    def self.sanitize(weights)
      weight_mapper.apply(weights)
    end

    def self.weight_mapper
      @weight_mapper ||= MLX::DSL.weight_map do
      end
    end

    def self.from_pretrained(path_or_repo, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      path = Pathname.new(path_or_repo.to_s)
      path = EncodecExample.snapshot_download(path_or_repo.to_s, python_bin: python_bin) unless path.exist?

      config_path = path.join("config.json")
      raise Errno::ENOENT, "Could not find #{config_path}" unless config_path.exist?

      raw_config = JSON.parse(File.binread(config_path))
      model = EncodecModel.new(EncodecConfig.from_hash(raw_config))

      weight_files = Dir.glob(path.join("*.npz").to_s).sort + Dir.glob(path.join("*.safetensors").to_s).sort
      unless weight_files.empty?
        weights = {}
        weight_files.each do |wf|
          MLX::Core.load(wf).to_a.each do |k, v|
            weights[k.to_s] = v
          end
        end
        begin
          model.load_weights(sanitize(weights).to_a, strict: false)
        rescue StandardError
          # Loading converted checkpoints is best-effort for this lightweight port.
        end
      end

      processor = lambda do |raw_audio|
        EncodecExample.preprocess_audio(
          raw_audio,
          sampling_rate: model.sampling_rate,
          chunk_length: model.chunk_length,
          chunk_stride: model.chunk_stride
        )
      end

      [model, processor]
    end
  end

  module_function

  def snapshot_download(repo_id, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    stdout, stderr, status = Open3.capture3(python_bin, SNAPSHOT_SCRIPT, repo_id.to_s)
    unless status.success?
      raise "Failed to download snapshot for #{repo_id}: #{stderr}"
    end

    Pathname.new(stdout.strip)
  end

  def preprocess_audio(raw_audio, sampling_rate: 24_000, chunk_length: nil, chunk_stride: nil)
    _ = sampling_rate

    sequences = raw_audio.is_a?(Array) ? raw_audio : [raw_audio]
    sequences = sequences.map do |audio|
      arr = Ops.as_array(audio)
      arr.shape.length == 1 ? MLX::Core.expand_dims(arr, 1) : arr
    end

    max_length = sequences.map { |seq| seq.shape[0] }.max
    if !chunk_length.nil?
      stride = chunk_stride || chunk_length
      max_length += chunk_length - (max_length % stride)
    end

    inputs = []
    masks = []
    sequences.each do |seq|
      length = seq.shape[0]
      mask = MLX::Core.ones([length], MLX::Core.bool_)
      diff = max_length - length
      if diff.positive?
        mask = MLX::Core.pad(mask, [[0, diff]])
        seq = MLX::Core.pad(seq, [[0, diff], [0, 0]])
      end
      inputs << seq
      masks << mask
    end

    [MLX::Core.stack(inputs, 0), MLX::Core.stack(masks, 0)]
  end
end
