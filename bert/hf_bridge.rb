# frozen_string_literal: true

require "json"
require "open3"
require "optparse"
require "pathname"

module BertExample
  class PythonBridge
    SCRIPT_DIR = Pathname.new(__dir__).join("python")
    SCRIPTS = {
      config: SCRIPT_DIR.join("config.py").to_s,
      tokenize: SCRIPT_DIR.join("tokenize_bridge.py").to_s,
      torch_forward: SCRIPT_DIR.join("torch_forward.py").to_s,
      convert: SCRIPT_DIR.join("convert.py").to_s
    }.freeze

    def initialize(python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @python_bin = python_bin
    end

    def config(model_name, config_path: nil)
      if !config_path.nil? && !config_path.to_s.empty?
        return JSON.parse(File.binread(config_path.to_s))
      end

      run_json(:config, model_name.to_s)
    end

    def tokenize(model_name, batch)
      run_json(:tokenize, model_name.to_s, JSON.generate(batch))
    end

    def torch_forward(model_name, batch)
      run_json(:torch_forward, model_name.to_s, JSON.generate(batch))
    end

    def convert_weights(model_name:, mlx_model:)
      run_json(:convert, model_name.to_s, mlx_model.to_s)
    end

    private

    def run_json(script_key, *args)
      script_path = SCRIPTS.fetch(script_key)
      stdout, stderr, status = Open3.capture3(@python_bin, script_path, *args)
      return JSON.parse(stdout) if status.success?

      raise RuntimeError, "python bridge failed (#{@python_bin} #{script_path} #{args.join(' ')}):\n#{stderr}"
    rescue JSON::ParserError => e
      raise RuntimeError, "python bridge returned invalid JSON: #{e.message}"
    end
  end
end

if $PROGRAM_NAME == __FILE__
  command = ARGV.shift
  if command.nil?
    warn "Usage: ruby bert/hf_bridge.rb <config|tokenize|torch_forward|convert> [options]"
    exit 1
  end

  options = {
    python_bin: ENV.fetch("PYTHON_BIN", "python3"),
    model: nil,
    config_path: nil,
    texts_json: nil,
    mlx_model: nil
  }

  parser = OptionParser.new do |opts|
    opts.on("--python-bin BIN", String, "Python binary") { |v| options[:python_bin] = v }
    opts.on("--model NAME", String, "Model name") { |v| options[:model] = v }
    opts.on("--config-path PATH", String, "Config JSON path") { |v| options[:config_path] = v }
    opts.on("--texts-json JSON", String, "JSON array of texts") { |v| options[:texts_json] = v }
    opts.on("--mlx-model PATH", String, "Output MLX model path for convert") { |v| options[:mlx_model] = v }
  end
  parser.parse!(ARGV)

  bridge = BertExample::PythonBridge.new(python_bin: options[:python_bin])

  payload = case command
  when "config"
    raise ArgumentError, "--model is required" if options[:model].nil? || options[:model].empty?
    bridge.config(options[:model], config_path: options[:config_path])
  when "tokenize"
    raise ArgumentError, "--model is required" if options[:model].nil? || options[:model].empty?
    raise ArgumentError, "--texts-json is required" if options[:texts_json].nil? || options[:texts_json].empty?
    texts = JSON.parse(options[:texts_json])
    bridge.tokenize(options[:model], texts)
  when "torch_forward"
    raise ArgumentError, "--model is required" if options[:model].nil? || options[:model].empty?
    raise ArgumentError, "--texts-json is required" if options[:texts_json].nil? || options[:texts_json].empty?
    texts = JSON.parse(options[:texts_json])
    bridge.torch_forward(options[:model], texts)
  when "convert"
    raise ArgumentError, "--model is required" if options[:model].nil? || options[:model].empty?
    raise ArgumentError, "--mlx-model is required" if options[:mlx_model].nil? || options[:mlx_model].empty?
    bridge.convert_weights(model_name: options[:model], mlx_model: options[:mlx_model])
  else
    raise ArgumentError, "unsupported command: #{command}"
  end

  puts JSON.generate(payload)
end
