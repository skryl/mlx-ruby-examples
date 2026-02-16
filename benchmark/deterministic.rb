# frozen_string_literal: true

module BenchmarkDeterministic
  module_function

  def install!
    return if @installed

    patch_random_uniform!
    patch_uniform!
    patch_normal!
    patch_truncated_normal!

    @installed = true
  end

  def patched?
    @installed == true
  end

  def tensor(shape:, dtype:, low:, high:, high_exclusive: false)
    normalized_shape = normalize_shape(shape)
    normalized_dtype = normalize_dtype(dtype)
    name = dtype_name(normalized_dtype)

    return MLX::Core.zeros(normalized_shape, normalized_dtype) if normalized_shape.any?(&:zero?)

    base = base_tensor(normalized_shape)
    scaled = scale_to_unit_interval(base)

    case name
    when /\Abool/
      cast_tensor(MLX::Core.greater(scaled, 0.5), normalized_dtype)
    when /\A(u?int)/
      int_tensor(
        scaled: scaled,
        dtype: normalized_dtype,
        low: low,
        high: high,
        high_exclusive: high_exclusive
      )
    else
      float_tensor(scaled: scaled, dtype: normalized_dtype, low: low, high: high)
    end
  end

  def tensor_like(value, low: -0.25, high: 0.25)
    return value unless value.respond_to?(:shape) && value.respond_to?(:dtype)

    tensor(
      shape: value.shape,
      dtype: value.dtype,
      low: low,
      high: high
    )
  end

  def reinitialize_module!(module_obj, low: -0.25, high: 0.25)
    return module_obj unless module_obj.respond_to?(:parameters) && module_obj.respond_to?(:update)

    flattened = MLX::Utils.tree_flatten(module_obj.parameters)
    rebuilt = flattened.map do |path, value|
      [path, tensor_like(value, low: low, high: high)]
    end

    module_obj.update(MLX::Utils.tree_unflatten(rebuilt))
    MLX::Core.eval(module_obj.parameters)
    module_obj
  end

  def patch_random_uniform!
    sc = MLX::Core.singleton_class
    return if sc.method_defined?(:__benchmark_original_random_uniform)

    sc.class_eval do
      alias_method :__benchmark_original_random_uniform, :random_uniform

      define_method(:random_uniform) do |*args, **kwargs|
        shape, low, high, dtype = BenchmarkDeterministic.parse_uniform_args(args, kwargs)

        BenchmarkDeterministic.tensor(
          shape: shape,
          dtype: dtype,
          low: low,
          high: high
        )
      end
    end
  end

  def patch_uniform!
    sc = MLX::Core.singleton_class
    return unless sc.method_defined?(:uniform)
    return if sc.method_defined?(:__benchmark_original_uniform)

    sc.class_eval do
      alias_method :__benchmark_original_uniform, :uniform

      define_method(:uniform) do |*args, **kwargs|
        shape, low, high, dtype = BenchmarkDeterministic.parse_uniform_args(args, kwargs)

        BenchmarkDeterministic.tensor(
          shape: shape,
          dtype: dtype,
          low: low,
          high: high
        )
      end
    end
  end

  def patch_normal!
    sc = MLX::Core.singleton_class
    return if sc.method_defined?(:__benchmark_original_normal)

    sc.class_eval do
      alias_method :__benchmark_original_normal, :normal

      define_method(:normal) do |*args, **kwargs|
        shape, mean, std, dtype = BenchmarkDeterministic.parse_normal_args(args, kwargs)
        low = mean.to_f - std.to_f
        high = mean.to_f + std.to_f

        BenchmarkDeterministic.tensor(
          shape: shape,
          dtype: dtype,
          low: low,
          high: high
        )
      end
    end
  end

  def patch_truncated_normal!
    sc = MLX::Core.singleton_class
    return unless sc.method_defined?(:truncated_normal)
    return if sc.method_defined?(:__benchmark_original_truncated_normal)

    sc.class_eval do
      alias_method :__benchmark_original_truncated_normal, :truncated_normal

      define_method(:truncated_normal) do |*args, **kwargs|
        low, high, shape, dtype = BenchmarkDeterministic.parse_truncated_normal_args(args, kwargs)

        BenchmarkDeterministic.tensor(
          shape: shape,
          dtype: dtype,
          low: low,
          high: high
        )
      end
    end
  end

  def parse_uniform_args(args, kwargs)
    shape = kwargs_value(kwargs, :shape)
    low = kwargs_value(kwargs, :low)
    high = kwargs_value(kwargs, :high)
    dtype = kwargs_value(kwargs, :dtype)

    if shape.nil?
      if args.length >= 3 && !shape_like?(args[0]) && shape_like?(args[2])
        low = args[0] if low.nil?
        high = args[1] if high.nil?
        shape = args[2]
        dtype = args[3] if dtype.nil?
      else
        shape = args[0]
        low = args[1] if low.nil?
        high = args[2] if high.nil?
        dtype = args[3] if dtype.nil?
      end
    end

    low = 0.0 if low.nil?
    high = 1.0 if high.nil?
    dtype = MLX::Core.float32 if dtype.nil?

    [shape, low, high, dtype]
  end

  def parse_normal_args(args, kwargs)
    shape = kwargs_value(kwargs, :shape)
    mean = kwargs_value(kwargs, :mean)
    mean = kwargs_value(kwargs, :loc) if mean.nil?
    std = kwargs_value(kwargs, :std)
    std = kwargs_value(kwargs, :scale) if std.nil?
    dtype = kwargs_value(kwargs, :dtype)

    if shape.nil?
      if args.length >= 3 && !shape_like?(args[0]) && shape_like?(args[2])
        mean = args[0] if mean.nil?
        std = args[1] if std.nil?
        shape = args[2]
        dtype = args[3] if dtype.nil?
      else
        shape = args[0]
        arg1 = args[1]
        arg2 = args[2]
        arg3 = args[3]

        if mean.nil? && std.nil? && dtype.nil? && dtype_value?(arg1) && arg2.nil?
          dtype = arg1
        else
          mean = arg1 if mean.nil?
          if std.nil? && dtype.nil? && dtype_value?(arg2)
            dtype = arg2
          else
            std = arg2 if std.nil?
          end
          dtype = arg3 if dtype.nil?
        end
      end
    end

    mean = 0.0 if mean.nil?
    std = 1.0 if std.nil?
    dtype = MLX::Core.float32 if dtype.nil?

    [shape, mean, std, dtype]
  end

  def parse_truncated_normal_args(args, kwargs)
    low = kwargs_value(kwargs, :low)
    high = kwargs_value(kwargs, :high)
    shape = kwargs_value(kwargs, :shape)
    dtype = kwargs_value(kwargs, :dtype)

    if shape.nil?
      if args.empty?
        raise ArgumentError, "truncated_normal requires a shape"
      end

      if shape_like?(args[0])
        shape = args[0]
        low = args[1] if low.nil?
        high = args[2] if high.nil?
        dtype = args[3] if dtype.nil?
      else
        low = args[0] if low.nil?
        high = args[1] if high.nil?
        shape = args[2]
        dtype = args[3] if dtype.nil?
      end
    end

    low = -0.02 if low.nil?
    high = 0.02 if high.nil?
    dtype = MLX::Core.float32 if dtype.nil?

    [low, high, shape, dtype]
  end

  def kwargs_value(kwargs, key)
    return nil unless kwargs
    return kwargs[key] if kwargs.key?(key)

    string_key = key.to_s
    return kwargs[string_key] if kwargs.key?(string_key)

    nil
  end

  def shape_like?(value)
    return false if value.nil?
    return true if value.is_a?(Array)
    return true if value.is_a?(Integer)
    return false if value.is_a?(Numeric)

    return true if value.respond_to?(:shape)

    return false unless value.respond_to?(:to_a)
    return false if value.is_a?(String)

    array_candidate = value.to_a
    array_candidate.is_a?(Array)
  rescue StandardError
    false
  end

  def normalize_shape(shape)
    return [] if shape.nil?
    return shape.map { |dim| dim.to_i } if shape.is_a?(Array)
    return [shape.to_i] if shape.is_a?(Numeric)

    if shape.respond_to?(:to_a) && !shape.is_a?(String)
      array_candidate = shape.to_a
      return array_candidate.map { |dim| dim.to_i } if array_candidate.is_a?(Array)
    end

    [shape.to_i]
  rescue StandardError
    [shape.to_i]
  end

  def dtype_name(dtype)
    if dtype.respond_to?(:name)
      dtype.name.to_s
    else
      dtype.to_s
    end
  end

  def dtype_value?(dtype)
    return true if dtype.nil?
    return true if dtype.is_a?(Symbol) || dtype.is_a?(String)

    dtype_class = MLX::Core.const_get(:Dtype)
    dtype.is_a?(dtype_class)
  rescue NameError
    false
  end

  def normalize_dtype(dtype)
    return MLX::Core.float32 if dtype.nil?
    return dtype if dtype_value?(dtype)

    MLX::Core.float32
  end

  def cast_tensor(array, dtype)
    normalized_dtype = normalize_dtype(dtype)
    return array if normalized_dtype.nil?

    array.astype(normalized_dtype)
  end

  def base_tensor(shape)
    size = shape.empty? ? 1 : shape.reduce(1, :*)
    base = MLX::Core.arange(0, size, 1, MLX::Core.float32)
    MLX::Core.reshape(base, shape)
  end

  def scale_to_unit_interval(base)
    wave = MLX::Core.sin(MLX::Core.add(base, 1.0))
    MLX::Core.multiply(MLX::Core.add(wave, 1.0), 0.5)
  end

  def int_tensor(scaled:, dtype:, low:, high:, high_exclusive:)
    low_i = low.to_i
    high_i = high.to_i
    span = if high_exclusive
      [high_i - low_i, 1].max
    else
      [high_i - low_i + 1, 1].max
    end
    shifted = MLX::Core.add(
      MLX::Core.floor(MLX::Core.multiply(scaled, span.to_f)),
      low_i.to_f
    )
    high_bound = high_exclusive ? (high_i - 1).to_f : high_i.to_f
    clipped = MLX::Core.clip(shifted, low_i.to_f, high_bound)
    cast_tensor(clipped, dtype)
  end

  def float_tensor(scaled:, dtype:, low:, high:)
    low_f = low.to_f
    high_f = high.to_f
    out = MLX::Core.add(low_f, MLX::Core.multiply(scaled, high_f - low_f))
    cast_tensor(out, dtype)
  end
end
