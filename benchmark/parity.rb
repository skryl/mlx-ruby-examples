# frozen_string_literal: true

require "digest"
require "json"
require "open3"
require "pathname"
require "shellwords"
require "tempfile"
require_relative "deterministic"

BenchmarkDeterministic.install! if ENV["MLX_BENCHMARK"] == "1"

module BenchmarkParity
  ROOT = Pathname.new(__dir__).join("..").expand_path.freeze
  SUBMODULE_ROOT = ROOT.join("mlx-examples").freeze
  PYTHON_RUNNER = ROOT.join("benchmark", "python", "run_model.py").freeze
  SCALE = 1_000_000.0
  NUMERIC_EPSILON = 0.001

  module_function

  def prime_backend!
    return if @backend_primed

    tensor = MLX::Core.array([0.0], MLX::Core.float32)
    MLX::Core.eval(MLX::Core.add(tensor, tensor))
    @backend_primed = true
  end

  def validate!(model_id:, inputs:, outputs:, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
    device = normalized_device(ENV["MLX_BENCHMARK_DEVICE"])
    ruby_inputs = canonicalize(inputs)
    ruby_outputs = canonicalize(outputs)
    ruby_input_signature = signature(ruby_inputs)
    ruby_output_signature = signature(ruby_outputs)

    python_payload = python_signatures(
      model_id: model_id,
      python_bin: python_bin,
      inputs_payload: ruby_inputs,
      device: device
    )
    py_input_signature = python_payload.fetch("input_signature")
    py_output_signature = python_payload.fetch("output_signature")
    py_inputs = python_payload.fetch("inputs")
    py_outputs = python_payload.fetch("outputs")

    input_shape_diff = first_shape_diff(ruby_inputs, py_inputs)
    input_diff = first_diff(ruby_inputs, py_inputs)
    output_shape_diff = first_shape_diff(ruby_outputs, py_outputs)
    output_diff = first_diff(ruby_outputs, py_outputs)
    summary = {
      "model" => model_id.to_s,
      "device" => device,
      "input_shape_match" => input_shape_diff.nil?,
      "input_content_match" => input_diff.nil?,
      "output_shape_match" => output_shape_diff.nil?,
      "output_content_match" => output_diff.nil?,
      "ruby_input_signature" => ruby_input_signature,
      "python_input_signature" => py_input_signature,
      "ruby_output_signature" => ruby_output_signature,
      "python_output_signature" => py_output_signature
    }
    puts "BENCHMARK_PARITY=#{JSON.generate(summary)}" if ENV["MLX_BENCHMARK"] == "1"

    return summary if summary["input_shape_match"] &&
      summary["input_content_match"] &&
      summary["output_shape_match"] &&
      summary["output_content_match"]

    if !summary["input_shape_match"] || !summary["input_content_match"]
      raise <<~MSG
        Benchmark input mismatch for #{model_id} (device=#{device}).
        ruby input signature:   #{ruby_input_signature}
        python input signature: #{py_input_signature}
        shape diff: #{format_diff(input_shape_diff)}
        content diff: #{format_diff(input_diff)}
      MSG
    end

    raise <<~MSG
      Benchmark output mismatch for #{model_id} (device=#{device}).
      ruby output signature:   #{ruby_output_signature}
      python output signature: #{py_output_signature}
      shape diff: #{format_diff(output_shape_diff)}
      content diff: #{format_diff(output_diff)}
    MSG
  end

  def signature(value)
    payload = canonicalize(value)
    json = JSON.generate(payload)
    Digest::SHA256.hexdigest(json)
  end

  def canonicalize(value)
    case value
    when Hash
      value.keys.map(&:to_s).sort.each_with_object({}) do |key, out|
        candidate = value[key]
        if candidate.nil? && value.key?(key.to_sym)
          candidate = value[key.to_sym]
        end
        out[key] = canonicalize(candidate)
      end
    when Array
      value.map { |item| canonicalize(item) }
    when Integer
      value
    when Float
      quantize_float(value)
    when Numeric
      quantize_float(value.to_f)
    when TrueClass, FalseClass, NilClass, String
      value
    else
      if mlx_array_like?(value)
        canonicalize(value.to_a)
      elsif value.respond_to?(:to_h)
        canonicalize(value.to_h)
      elsif value.respond_to?(:to_a) && !value.is_a?(String)
        array_candidate = value.to_a
        return canonicalize(array_candidate) if array_candidate.is_a?(Array)
        value.to_s
      else
        value.to_s
      end
    end
  end

  def quantize_float(number)
    ((number * SCALE).round / SCALE).to_f
  end

  def mlx_array_like?(value)
    value.respond_to?(:to_a) && value.respond_to?(:shape) && !value.is_a?(Array)
  end

  def python_signatures(model_id:, python_bin:, inputs_payload: nil, device: "gpu")
    python_cmd = Shellwords.split(python_bin.to_s)
    raise ArgumentError, "PYTHON_BIN must not be empty" if python_cmd.empty?

    command = python_cmd + [
      PYTHON_RUNNER.to_s,
      "--mlx-examples", SUBMODULE_ROOT.to_s,
      "--model", model_id.to_s,
      "--device", normalized_device(device),
      "--signatures-only",
      "--parity-payload"
    ]

    input_file = nil
    if inputs_payload
      input_file = Tempfile.new(["benchmark_inputs_", ".json"], ROOT.to_s)
      input_file.write(JSON.generate(inputs_payload))
      input_file.flush
      command += ["--inputs-json-file", input_file.path]
    end

    stdout, stderr, status = Open3.capture3(*command, chdir: ROOT.to_s)
    unless status.success?
      raise <<~MSG
        Python parity command failed for #{model_id}.
        command: #{command.join(' ')}
        stdout:
        #{stdout}
        stderr:
        #{stderr}
      MSG
    end

    line = stdout.lines.reverse.find { |entry| !entry.strip.empty? }
    raise "Python parity output missing JSON payload for #{model_id}" if line.nil?

    JSON.parse(line)
  ensure
    input_file&.close!
  end

  def normalized_device(raw)
    device = raw.to_s.strip.downcase
    return "cpu" if device == "cpu"

    "gpu"
  end

  def first_shape_diff(left, right, path = "$")
    if left.is_a?(Hash) && right.is_a?(Hash)
      keys = (left.keys + right.keys).uniq.sort
      keys.each do |key|
        return [path, left.keys.sort, right.keys.sort] unless left.key?(key) && right.key?(key)

        sub_diff = first_shape_diff(left[key], right[key], "#{path}.#{key}")
        return sub_diff unless sub_diff.nil?
      end
      return nil
    end

    if left.is_a?(Array) && right.is_a?(Array)
      return [path, left.length, right.length] unless left.length == right.length

      left.each_with_index do |item, index|
        sub_diff = first_shape_diff(item, right[index], "#{path}[#{index}]")
        return sub_diff unless sub_diff.nil?
      end
      return nil
    end

    return nil if same_shape_leaf?(left, right)

    [path, left.class.to_s, right.class.to_s]
  end

  def same_shape_leaf?(left, right)
    return true if left.is_a?(Numeric) && right.is_a?(Numeric)
    return true if left.nil? && right.nil?
    return true if [left, right].all? { |value| value.is_a?(TrueClass) || value.is_a?(FalseClass) }
    return true if left.is_a?(String) && right.is_a?(String)

    left.class == right.class
  end

  def first_diff(left, right, path = "$")
    if left.is_a?(Numeric) && right.is_a?(Numeric)
      return nil if (left.to_f - right.to_f).abs <= NUMERIC_EPSILON
    end

    return nil if left == right

    if left.is_a?(Hash) && right.is_a?(Hash)
      keys = (left.keys + right.keys).uniq.sort
      keys.each do |key|
        return [path, left, right] unless left.key?(key) && right.key?(key)

        sub_diff = first_diff(left[key], right[key], "#{path}.#{key}")
        return sub_diff unless sub_diff.nil?
      end
      return nil
    end

    if left.is_a?(Array) && right.is_a?(Array)
      return [path, left.length, right.length] unless left.length == right.length

      left.each_with_index do |item, index|
        sub_diff = first_diff(item, right[index], "#{path}[#{index}]")
        return sub_diff unless sub_diff.nil?
      end
      return nil
    end

    [path, left, right]
  end

  def format_diff(diff)
    return "none" if diff.nil?

    path, left, right = diff
    "#{path} ruby=#{left.inspect} python=#{right.inspect}"
  end
end
