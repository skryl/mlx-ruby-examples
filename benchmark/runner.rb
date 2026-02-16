# frozen_string_literal: true

require "digest"
require "fileutils"
require "json"
require "open3"
require "pathname"
require "rbconfig"
require "shellwords"
require "time"

module BenchmarkRunner
  ROOT = Pathname.new(__dir__).join("..").expand_path.freeze
  SUBMODULE_ROOT = ROOT.join("mlx-examples").freeze
  PYTHON_RUNNER = ROOT.join("benchmark", "python", "run_model.py").freeze
  BENCH_ROOT = ROOT.join("tmp", "benchmark").freeze
  LOG_DIR = BENCH_ROOT.join("logs").freeze
  RESULTS_DIR = BENCH_ROOT.join("results").freeze
  DEVICE_ORDER = %w[cpu gpu].freeze
  RUBY_DEVICE_REQUIRE = {
    "cpu" => ROOT.join("benchmark", "ruby", "force_cpu.rb").freeze,
    "gpu" => ROOT.join("benchmark", "ruby", "force_gpu.rb").freeze
  }.freeze
  RUBY_WARMUP_WRAPPER = ROOT.join("benchmark", "ruby", "run_with_dryrun.rb").freeze
  PREFERRED_PYTHON_CANDIDATES = %w[
    python3.12
    python3.11
    python3.10
    python3
    python
  ].freeze

  MODEL_SPECS = [
    {
      id: "bert",
      ruby_script: "bert/test.rb",
      python_marker: "bert/model.py",
      python_requirements: ["bert/requirements.txt"]
    },
    {
      id: "cifar",
      ruby_script: "cifar/test.rb",
      python_marker: "cifar/resnet.py",
      python_requirements: ["cifar/requirements.txt"]
    },
    {
      id: "clip",
      ruby_script: "clip/test.rb",
      python_marker: "clip/model.py",
      python_requirements: ["clip/requirements.txt"]
    },
    {
      id: "cvae",
      ruby_script: "cvae/test.rb",
      python_marker: "cvae/vae.py",
      python_requirements: ["cvae/requirements.txt"]
    },
    {
      id: "encodec",
      ruby_script: "encodec/test.rb",
      python_marker: "encodec/encodec.py",
      python_requirements: ["encodec/requirements.txt"]
    },
    {
      id: "flux",
      ruby_script: "flux/test.rb",
      python_marker: "flux/flux/model.py",
      python_requirements: ["flux/requirements.txt"]
    },
    {
      id: "gcn",
      ruby_script: "gcn/test.rb",
      python_marker: "gcn/gcn.py",
      python_requirements: ["gcn/requirements.txt"]
    },
    {
      id: "llava",
      ruby_script: "llava/test.rb",
      python_marker: "llava/llava.py",
      python_requirements: ["llava/requirements.txt"]
    },
    {
      id: "llms/gguf_llm",
      ruby_script: "llms/gguf_llm/test.rb",
      python_marker: "llms/gguf_llm/models.py",
      python_requirements: []
    },
    {
      id: "llms/llama",
      ruby_script: "llms/llama/test.rb",
      python_marker: "llms/llama/llama.py",
      python_requirements: []
    },
    {
      id: "llms/mistral",
      ruby_script: "llms/mistral/test.rb",
      python_marker: "llms/mistral/mistral.py",
      python_requirements: []
    },
    {
      id: "llms/mixtral",
      ruby_script: "llms/mixtral/test.rb",
      python_marker: "llms/mixtral/mixtral.py",
      python_requirements: []
    },
    {
      id: "llms/speculative_decoding",
      ruby_script: "llms/speculative_decoding/test.rb",
      python_marker: "llms/speculative_decoding/model.py",
      python_requirements: []
    },
    {
      id: "lora",
      ruby_script: "lora/test.rb",
      python_marker: "lora/models.py",
      python_requirements: ["lora/requirements.txt"]
    },
    {
      id: "mnist",
      ruby_script: "mnist/test.rb",
      python_marker: "mnist/main.py",
      python_requirements: ["mnist/requirements.txt"]
    },
    {
      id: "musicgen",
      ruby_script: "musicgen/test.rb",
      python_marker: "musicgen/musicgen.py",
      python_requirements: ["musicgen/requirements.txt"]
    },
    {
      id: "normalizing_flow",
      ruby_script: "normalizing_flow/test.rb",
      python_marker: "normalizing_flow/flows.py",
      python_requirements: ["normalizing_flow/requirements.txt"]
    },
    {
      id: "segment_anything",
      ruby_script: "segment_anything/test.rb",
      python_marker: "segment_anything/segment_anything/image_encoder.py",
      python_requirements: ["segment_anything/requirements.txt"]
    },
    {
      id: "speechcommands",
      ruby_script: "speechcommands/test.rb",
      python_marker: "speechcommands/kwt.py",
      python_requirements: ["speechcommands/requirements.txt"]
    },
    {
      id: "stable_diffusion",
      ruby_script: "stable_diffusion/test.rb",
      python_marker: "stable_diffusion/stable_diffusion/unet.py",
      python_requirements: ["stable_diffusion/requirements.txt"]
    },
    {
      id: "t5",
      ruby_script: "t5/test.rb",
      python_marker: "t5/t5.py",
      python_requirements: ["t5/requirements.txt"]
    },
    {
      id: "transformer_lm",
      ruby_script: "transformer_lm/test.rb",
      python_marker: "transformer_lm/main.py",
      python_requirements: ["transformer_lm/requirements.txt"]
    },
    {
      id: "whisper",
      ruby_script: "whisper/test.rb",
      python_marker: "whisper/mlx_whisper/whisper.py",
      python_requirements: []
    }
  ].freeze

  EXTRA_PYTHON_PACKAGES = %w[
    mlx
    numpy
    transformers
    huggingface_hub
    tqdm
    sentencepiece
    protobuf
    Pillow
    tiktoken
  ].freeze

  module_function

  def run(mode:)
    validate_mode!(mode)

    selected_specs = select_models(MODEL_SPECS, ENV["MODELS"])
    ensure_submodule_present!(selected_specs)

    runs = integer_env("RUNS", default: 1, min: 1)
    warmup = integer_env("WARMUP", default: 0, min: 0)
    timeout = integer_env("BENCH_TIMEOUT", default: 900, min: 1)

    FileUtils.mkdir_p(BENCH_ROOT)
    FileUtils.mkdir_p(LOG_DIR)
    FileUtils.mkdir_p(RESULTS_DIR)

    python_base = resolve_python_base_command
    python_base_version = python_command_version(python_base)
    python_base_abiflags = python_command_abiflags(python_base)
    venv_python = setup_python_environment!(python_base: python_base, selected_specs: selected_specs)
    ruby_bin = RbConfig.ruby

    puts "Benchmark mode: #{mode}"
    puts "Models: #{selected_specs.map { |s| s.fetch(:id) }.join(', ')}"
    puts "Python base command: #{python_base.join(' ')}#{python_base_version ? " (#{python_base_version.join('.')}#{python_base_abiflags})" : ''}"
    puts "Python executable: #{venv_python}"
    puts "Ruby executable: #{ruby_bin}"
    puts "Warmup iterations: #{warmup}"
    puts "Measured iterations: #{runs}"
    puts

    python_results = run_python_benchmarks(
      specs: selected_specs,
      python_bin: venv_python,
      runs: runs,
      warmup: warmup,
      timeout: timeout
    )

    ruby_results = run_ruby_benchmarks(
      specs: selected_specs,
      ruby_bin: ruby_bin,
      python_bin: venv_python,
      mode: mode,
      runs: runs,
      warmup: warmup,
      timeout: timeout
    )

    report = build_report(
      mode: mode,
      runs: runs,
      warmup: warmup,
      specs: selected_specs,
      python_results: python_results,
      ruby_results: ruby_results
    )

    emit_report(report)
    write_report(report)

    failures = report.fetch(:rows).select { |row| critical_row_failure?(row) }
    partial_failures = report.fetch(:rows).reject { |row| failures.include?(row) }.select { |row| row.fetch(:python_status) != 0 || row.fetch(:ruby_status) != 0 }

    if failures.empty? && partial_failures.empty?
      return
    end

    unless partial_failures.empty?
      warn
      warn "benchmark completed with partial device failures:"
      partial_failures.each do |row|
        py = row.fetch(:python)
        rb = row.fetch(:ruby)
        warn "  - #{row.fetch(:id)} (py=#{status_pair(py)}, rb=#{status_pair(rb)})"
      end
    end

    return if failures.empty?

    warn
    warn "benchmark hard failures detected (all devices failed for at least one language):"
    failures.each do |row|
      py = row.fetch(:python)
      rb = row.fetch(:ruby)
      warn "  - #{row.fetch(:id)} (py=#{status_pair(py)}, rb=#{status_pair(rb)})"
    end

    exit 1
  end

  def validate_mode!(mode)
    return if %w[dsl no_dsl].include?(mode)

    raise ArgumentError, "Unsupported benchmark mode: #{mode.inspect}"
  end

  def integer_env(key, default:, min:)
    raw = ENV[key]
    return default if raw.nil? || raw.strip.empty?

    value = Integer(raw, 10)
    raise ArgumentError, "#{key} must be >= #{min}" if value < min

    value
  rescue ArgumentError
    raise ArgumentError, "Invalid integer for #{key}: #{raw.inspect}"
  end

  def select_models(specs, raw_filter)
    return specs if raw_filter.nil? || raw_filter.strip.empty?

    wanted = raw_filter.split(",").map(&:strip).reject(&:empty?)
    unknown = wanted - specs.map { |s| s.fetch(:id) }
    unless unknown.empty?
      raise ArgumentError, "Unknown model(s) in MODELS: #{unknown.join(', ')}"
    end

    specs.select { |spec| wanted.include?(spec.fetch(:id)) }
  end

  def ensure_submodule_present!(selected_specs)
    unless SUBMODULE_ROOT.join("README.md").exist?
      abort <<~MSG
        Missing mlx-examples submodule contents at #{SUBMODULE_ROOT}.
        Run: git submodule update --init --recursive mlx-examples
      MSG
    end

    missing_markers = selected_specs.filter_map do |spec|
      marker = SUBMODULE_ROOT.join(spec.fetch(:python_marker))
      marker.to_s unless marker.exist?
    end

    return if missing_markers.empty?

    warn "mlx-examples submodule is missing required files."
    missing_markers.each { |path| warn "  - #{path}" }
    abort "Ensure submodule contents are populated before benchmarking."
  end

  def resolve_python_base_command
    env_python = ENV["PYTHON_BIN"]
    unless env_python.nil? || env_python.strip.empty?
      explicit = Shellwords.split(env_python)
      explicit_version = python_command_version(explicit)
      explicit_abiflags = python_command_abiflags(explicit)
      if explicit_version && explicit_version[0] == 3 && explicit_version[1] > 12
        warn "Warning: PYTHON_BIN resolves to Python #{explicit_version.join('.')}."
        warn "The 'mlx' Python package may be unavailable for this version."
        warn "If setup fails, set PYTHON_BIN to a Python 3.12/3.11 executable."
      end
      if explicit_abiflags.include?("t")
        warn "Warning: PYTHON_BIN appears to be a free-threaded Python build (abiflags='#{explicit_abiflags}')."
        warn "MLX wheels are typically not published for cp* t ABIs (e.g., cp313t)."
      end
      return explicit
    end

    candidates = []

    bin_env = ROOT.join("bin", "env")
    if bin_env.exist?
      PREFERRED_PYTHON_CANDIDATES.each do |candidate|
        command = [bin_env.to_s, candidate]
        candidates << command if executable_python?(command)
      end
    end

    PREFERRED_PYTHON_CANDIDATES.each do |candidate|
      command = [candidate]
      candidates << command if executable_python?(command)
    end

    selected = select_best_python(candidates)
    return selected if selected

    abort <<~MSG
      Could not locate a usable Python interpreter.
      Set PYTHON_BIN to a Python 3.12 or 3.11 executable and re-run benchmark.
    MSG
  end

  def setup_python_environment!(python_base:, selected_specs:)
    venv_dir = venv_dir_for(python_base)
    venv_python = venv_dir.join("bin", "python")

    if venv_dir.exist? && !venv_python.exist?
      puts "Removing incomplete Python venv at #{venv_dir}"
      FileUtils.rm_rf(venv_dir)
    end

    unless venv_python.exist?
      puts "Creating Python venv in #{venv_dir}"
      ensure_command_success!(python_base + ["-m", "venv", venv_dir.to_s], label: "create venv", timeout: 600)
    end

    digest = dependency_digest(selected_specs)
    stamp_file = venv_dir.join(".python_deps.sha256")
    up_to_date = stamp_file.exist? && stamp_file.read.strip == digest

    if up_to_date
      puts "Python dependencies are up to date."
      return venv_python.to_s
    end

    puts "Installing Python dependencies for benchmarks"
    ensure_command_success!(
      [venv_python.to_s, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
      label: "upgrade pip",
      timeout: 900
    )

    requirement_files(selected_specs).each do |req_file|
      ensure_command_success!(
        [venv_python.to_s, "-m", "pip", "install", "-r", req_file.to_s],
        label: "install #{req_file.relative_path_from(SUBMODULE_ROOT)}",
        timeout: 900
      )
    end

    ensure_command_success!(
      [venv_python.to_s, "-m", "pip", "install", *EXTRA_PYTHON_PACKAGES],
      label: "install shared benchmark dependencies",
      timeout: 900
    )

    stamp_file.write("#{digest}\n")
    venv_python.to_s
  end

  def executable_python?(command)
    result = run_capture(command + ["-c", "import sys"], chdir: ROOT, env: {}, timeout: 30)
    result.fetch(:status).zero?
  end

  def python_command_version(command)
    result = run_capture(
      command + ["-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
      chdir: ROOT,
      env: {},
      timeout: 30
    )
    return nil unless result.fetch(:status).zero?

    version = result.fetch(:stdout).strip
    major, minor = version.split(".").map { |x| Integer(x, 10) }
    [major, minor]
  rescue StandardError
    nil
  end

  def python_command_abiflags(command)
    result = run_capture(
      command + ["-c", "import sys; print(getattr(sys, 'abiflags', '') or '')"],
      chdir: ROOT,
      env: {},
      timeout: 30
    )
    return "" unless result.fetch(:status).zero?

    result.fetch(:stdout).strip
  rescue StandardError
    ""
  end

  def select_best_python(candidates)
    inspected = candidates.uniq.map do |command|
      version = python_command_version(command)
      next if version.nil?
      abiflags = python_command_abiflags(command)
      { command: command, version: version, abiflags: abiflags }
    end.compact
    return nil if inspected.empty?

    non_free_threaded = inspected.reject { |entry| entry.fetch(:abiflags).include?("t") }
    inspected = non_free_threaded unless non_free_threaded.empty?

    compatible = inspected.select { |entry| entry.fetch(:version) == [3, 12] }
    compatible = inspected.select { |entry| entry.fetch(:version) == [3, 11] } if compatible.empty?
    compatible = inspected.select { |entry| entry.fetch(:version)[0] == 3 && entry.fetch(:version)[1] <= 12 } if compatible.empty?

    pool = compatible.empty? ? inspected : compatible
    pool.max_by { |entry| entry.fetch(:version) }.fetch(:command)
  end

  def venv_dir_for(python_base)
    digest = Digest::SHA256.hexdigest(python_base.join("\0"))[0, 12]
    BENCH_ROOT.join(".venv_#{digest}")
  end

  def requirement_files(_selected_specs)
    []
  end

  def dependency_digest(selected_specs)
    commit = capture_stdout(["git", "-C", SUBMODULE_ROOT.to_s, "rev-parse", "HEAD"]).strip
    req_payload = requirement_files(selected_specs).map do |path|
      "#{path.relative_path_from(SUBMODULE_ROOT)}\n#{path.read}"
    end

    payload = [
      "submodule=#{commit}",
      "specs=#{selected_specs.map { |s| s.fetch(:id) }.join(',')}",
      "requirements:\n#{req_payload.join("\n---\n")}",
      "extras=#{EXTRA_PYTHON_PACKAGES.join(',')}"
    ].join("\n")

    Digest::SHA256.hexdigest(payload)
  end

  def run_python_benchmarks(specs:, python_bin:, runs:, warmup:, timeout:)
    puts "=== Python Benchmarks ==="
    results = specs.each_with_object({}) { |spec, out| out[spec.fetch(:id)] = {} }

    DEVICE_ORDER.each do |device|
      puts "Device: #{device}"
      specs.each do |spec|
        id = spec.fetch(:id)
        command = [
          python_bin,
          PYTHON_RUNNER.to_s,
          "--mlx-examples", SUBMODULE_ROOT.to_s,
          "--model", id,
          "--device", device,
          "--warmup", warmup.to_s,
          "--runs", runs.to_s
        ]

        log_file = LOG_DIR.join("python_#{device}_#{id.tr('/', '__')}_run_1.log")
        command_result = run_capture(command, chdir: ROOT, env: {}, timeout: timeout)
        write_log(log_file: log_file, command: command, result: command_result)
        result = format_python_result(command_result, log_file: log_file)

        results[id][device] = result
        puts format("PY  %-4s %-28s %s", device.upcase, id, summarize_result(result))
      end
    end

    puts
    results
  end

  def run_ruby_benchmarks(specs:, ruby_bin:, python_bin:, mode:, runs:, warmup:, timeout:)
    puts "=== Ruby Benchmarks (#{mode}) ==="
    results = specs.each_with_object({}) { |spec, out| out[spec.fetch(:id)] = {} }

    DEVICE_ORDER.each do |device|
      puts "Device: #{device}"
      require_path = RUBY_DEVICE_REQUIRE.fetch(device)

      specs.each do |spec|
        id = spec.fetch(:id)
        script_path = mode == "no_dsl" ? File.join("no_dsl", spec.fetch(:ruby_script)) : spec.fetch(:ruby_script)
        absolute_script = ROOT.join(script_path)

        unless absolute_script.exist?
          result = {
            status: 1,
            seconds: nil,
            samples: [],
            iterations: [{ log: nil, status: 1, stderr: "Missing script: #{absolute_script}", stdout: "", elapsed: 0.0 }],
            parity: nil
          }
          results[id][device] = result
          puts format("RB  %-4s %-28s %s", device.upcase, id, summarize_result(result))
          next
        end

        command = [ruby_bin, "-r", require_path.to_s, RUBY_WARMUP_WRAPPER.to_s, script_path]
        result = benchmark_iterations(
          phase: "ruby_#{mode}_#{device}",
          id: id,
          command: command,
          chdir: ROOT,
          env: {
            "PYTHON_BIN" => python_bin,
            "MLX_BENCHMARK" => "1",
            "MLX_BENCHMARK_DEVICE" => device
          },
          runs: runs,
          warmup: warmup,
          timeout: timeout,
          sample_parser: method(:parse_ruby_benchmark_seconds),
          parity_parser: method(:parse_ruby_parity)
        )

        results[id][device] = result
        puts format("RB  %-4s %-28s %s", device.upcase, id, summarize_result(result))
      end
    end

    puts
    results
  end

  def benchmark_iterations(phase:, id:, command:, chdir:, env:, runs:, warmup:, timeout:, sample_parser:, parity_parser: nil)
    samples = []
    iterations = []
    status = 0
    parity = nil

    total = warmup + runs
    total.times do |index|
      is_warmup = index < warmup
      iter = is_warmup ? (index + 1) : (index - warmup + 1)
      kind = is_warmup ? "warmup" : "run"

      log_file = LOG_DIR.join("#{phase}_#{id.tr('/', '__')}_#{kind}_#{iter}.log")
      result = run_capture(command, chdir: chdir, env: env, timeout: timeout)
      write_log(log_file: log_file, command: command, result: result)

      payload = result.merge(log: log_file.to_s, kind: kind, iteration: iter)
      measured_seconds = nil

      if result.fetch(:status).zero?
        begin
          measured_seconds = sample_parser.call(result.fetch(:stdout))
          if parity_parser
            payload[:parity] = parity_parser.call(result.fetch(:stdout))
            if !is_warmup && payload[:parity].nil?
              raise ArgumentError, "Ruby benchmark did not emit BENCHMARK_PARITY line"
            end
          end
        rescue StandardError => e
          payload[:status] = 1
          payload[:stderr] = [
            result.fetch(:stderr),
            "Failed to parse benchmark sample: #{e.class}: #{e.message}",
            "STDOUT:",
            result.fetch(:stdout)
          ].join("\n")
        end
      end

      iterations << payload

      unless payload.fetch(:status).zero?
        status = payload.fetch(:status)
        break
      end

      unless is_warmup
        samples << measured_seconds
        parity = payload[:parity] if payload.key?(:parity)
      end
    end

    {
      status: status,
      seconds: samples.empty? ? nil : samples.sum / samples.length,
      samples: samples,
      iterations: iterations,
      parity: parity
    }
  end

  def format_python_result(command_result, log_file:)
    payload = command_result.merge(log: log_file.to_s, kind: "run", iteration: 1)

    unless command_result.fetch(:status).zero?
      return {
        status: command_result.fetch(:status),
        seconds: nil,
        samples: [],
        iterations: [payload]
      }
    end

    data = parse_json_payload(command_result.fetch(:stdout))
    samples = Array(data.fetch("samples")).map { |value| Float(value) }
    seconds = Float(data.fetch("seconds"))

    {
      status: 0,
      seconds: seconds,
      samples: samples,
      iterations: [payload]
    }
  rescue StandardError => e
    payload[:status] = 1
    payload[:stderr] = [
      command_result.fetch(:stderr),
      "Failed to parse Python benchmark JSON: #{e.class}: #{e.message}",
      "STDOUT:",
      command_result.fetch(:stdout)
    ].join("\n")

    {
      status: 1,
      seconds: nil,
      samples: [],
      iterations: [payload]
    }
  end

  def parse_json_payload(stdout)
    line = stdout.lines.reverse.find { |entry| !entry.strip.empty? }
    raise ArgumentError, "Python benchmark produced no JSON output" if line.nil?

    JSON.parse(line)
  end

  def parse_ruby_benchmark_seconds(stdout)
    line = stdout.lines.reverse.find { |entry| entry.include?("BENCHMARK_SECONDS=") }
    raise ArgumentError, "Ruby benchmark did not emit BENCHMARK_SECONDS line" if line.nil?

    raw = line.split("=", 2).last
    raise ArgumentError, "Ruby BENCHMARK_SECONDS line is malformed" if raw.nil?

    Float(raw.strip)
  end

  def parse_ruby_parity(stdout)
    line = stdout.lines.reverse.find { |entry| entry.include?("BENCHMARK_PARITY=") }
    return nil if line.nil?

    raw = line.split("=", 2).last
    return nil if raw.nil?

    JSON.parse(raw.strip)
  rescue JSON::ParserError
    nil
  end

  def summarize_result(result)
    status = result.fetch(:status)
    if status.zero?
      avg = result.fetch(:seconds)
      total = total_seconds(avg, result.fetch(:samples))
      if total.nil?
        "#{format_seconds(avg)}s avg"
      else
        "#{format_seconds(avg)}s avg / #{format_seconds(total)}s total"
      end
    else
      last = result.fetch(:iterations).last
      log = last && last[:log]
      "FAILED (status #{status})#{log ? " [log: #{log}]" : ""}"
    end
  end

  def build_report(mode:, runs:, warmup:, specs:, python_results:, ruby_results:)
    rows = specs.map do |spec|
      id = spec.fetch(:id)
      py_cpu = device_result(python_results, id, "cpu")
      py_gpu = device_result(python_results, id, "gpu")
      rb_cpu = device_result(ruby_results, id, "cpu")
      rb_gpu = device_result(ruby_results, id, "gpu")

      py_cpu_s = py_cpu.fetch(:seconds)
      py_gpu_s = py_gpu.fetch(:seconds)
      rb_cpu_s = rb_cpu.fetch(:seconds)
      rb_gpu_s = rb_gpu.fetch(:seconds)

      parity_cpu = rb_cpu[:parity]
      parity_gpu = rb_gpu[:parity]

      {
        id: id,
        python: {
          "cpu" => summarize_device_result(py_cpu),
          "gpu" => summarize_device_result(py_gpu)
        },
        ruby: {
          "cpu" => summarize_device_result(rb_cpu).merge(parity: parity_cpu),
          "gpu" => summarize_device_result(rb_gpu).merge(parity: parity_gpu)
        },
        ratios: {
          python_cpu_per_gpu: safe_ratio(py_cpu_s, py_gpu_s),
          ruby_cpu_per_gpu: safe_ratio(rb_cpu_s, rb_gpu_s),
          ruby_per_python_cpu: safe_ratio(rb_cpu_s, py_cpu_s),
          ruby_per_python_gpu: safe_ratio(rb_gpu_s, py_gpu_s)
        },
        parity: {
          input_shape: parity_pair(parity_cpu, parity_gpu, "input_shape_match"),
          input_content: parity_pair(parity_cpu, parity_gpu, "input_content_match"),
          output_shape: parity_pair(parity_cpu, parity_gpu, "output_shape_match"),
          output_content: parity_pair(parity_cpu, parity_gpu, "output_content_match")
        },
        python_status: aggregate_status(py_cpu, py_gpu),
        ruby_status: aggregate_status(rb_cpu, rb_gpu)
      }
    end

    {
      generated_at: Time.now.utc.iso8601,
      mode: mode,
      runs: runs,
      warmup: warmup,
      rows: rows
    }
  end

  def device_result(results, id, device)
    result = results.fetch(id, {}).fetch(device, nil)
    return result unless result.nil?

    {
      status: 1,
      seconds: nil,
      samples: [],
      iterations: [{ log: nil, status: 1, stderr: "Missing benchmark result for #{id} (#{device})", stdout: "", elapsed: 0.0 }],
      parity: nil
    }
  end

  def summarize_device_result(result)
    seconds = result.fetch(:seconds)
    samples = result.fetch(:samples)

    {
      seconds: seconds,
      total_seconds: total_seconds(seconds, samples),
      measured_runs: samples.length,
      status: result.fetch(:status),
      log: result.fetch(:iterations).last&.fetch(:log, nil),
      samples: samples
    }
  end

  def total_seconds(seconds, samples)
    return nil if seconds.nil?
    return nil if samples.nil? || samples.empty?

    seconds * samples.length
  end

  def safe_ratio(numerator, denominator)
    return nil if numerator.nil? || denominator.nil? || denominator <= 0.0

    numerator / denominator
  end

  def aggregate_status(*results)
    statuses = results.flatten.compact.map { |entry| entry.fetch(:status).to_i }
    statuses.find { |code| code != 0 } || 0
  end

  def parity_pair(cpu_payload, gpu_payload, key)
    {
      "cpu" => parity_value(cpu_payload, key),
      "gpu" => parity_value(gpu_payload, key)
    }
  end

  def parity_value(payload, key)
    return nil unless payload.is_a?(Hash)
    return payload[key] unless payload[key].nil?

    payload[key.to_sym]
  end

  def emit_report(report)
    puts "=== Final Benchmark Table ==="
    puts "Times shown as average per measured iteration and total across measured iterations."
    puts "| model | py_cpu_avg_s | py_cpu_total_s | py_gpu_avg_s | py_gpu_total_s | py_cpu/gpu | rb_cpu_avg_s | rb_cpu_total_s | rb_gpu_avg_s | rb_gpu_total_s | rb_cpu/gpu | rb/py_cpu | rb/py_gpu | in_shape (cpu/gpu) | in_content (cpu/gpu) | out_shape (cpu/gpu) | out_content (cpu/gpu) |"
    puts "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |"

    report.fetch(:rows).each do |row|
      py = row.fetch(:python)
      rb = row.fetch(:ruby)
      ratios = row.fetch(:ratios)
      parity = row.fetch(:parity)
      puts [
        "| #{row.fetch(:id)}",
        format_seconds(py.fetch("cpu").fetch(:seconds)),
        format_seconds(py.fetch("cpu").fetch(:total_seconds)),
        format_seconds(py.fetch("gpu").fetch(:seconds)),
        format_seconds(py.fetch("gpu").fetch(:total_seconds)),
        format_ratio(ratios.fetch(:python_cpu_per_gpu)),
        format_seconds(rb.fetch("cpu").fetch(:seconds)),
        format_seconds(rb.fetch("cpu").fetch(:total_seconds)),
        format_seconds(rb.fetch("gpu").fetch(:seconds)),
        format_seconds(rb.fetch("gpu").fetch(:total_seconds)),
        format_ratio(ratios.fetch(:ruby_cpu_per_gpu)),
        format_ratio(ratios.fetch(:ruby_per_python_cpu)),
        format_ratio(ratios.fetch(:ruby_per_python_gpu)),
        format_check_pair(parity.fetch(:input_shape)),
        format_check_pair(parity.fetch(:input_content)),
        format_check_pair(parity.fetch(:output_shape)),
        format_check_pair(parity.fetch(:output_content))
      ].join(" | ") + " |"
    end
  end

  def format_seconds(value)
    return "-" if value.nil?

    if value.abs >= 1.0
      format("%.3f", value)
    elsif value.abs >= 0.01
      format("%.4f", value)
    else
      format("%.6f", value)
    end
  end

  def format_ratio(value)
    return "-" if value.nil?

    format("%.2fx", value)
  end

  def format_check_pair(pair)
    cpu = check_symbol(pair.fetch("cpu", nil))
    gpu = check_symbol(pair.fetch("gpu", nil))
    "#{cpu}/#{gpu}"
  end

  def check_symbol(value)
    return "-" if value.nil?

    value ? "✓" : "✗"
  end

  def format_status_pair(language_payload)
    cpu = language_payload.fetch("cpu").fetch(:status).to_i
    gpu = language_payload.fetch("gpu").fetch(:status).to_i
    "#{cpu}/#{gpu}"
  end

  def status_pair(language_payload)
    cpu = language_payload.fetch("cpu").fetch(:status).to_i
    gpu = language_payload.fetch("gpu").fetch(:status).to_i
    "#{cpu}/#{gpu}"
  end

  def critical_row_failure?(row)
    py = row.fetch(:python)
    rb = row.fetch(:ruby)
    py_all_failed = py.fetch("cpu").fetch(:status).to_i != 0 && py.fetch("gpu").fetch(:status).to_i != 0
    rb_all_failed = rb.fetch("cpu").fetch(:status).to_i != 0 && rb.fetch("gpu").fetch(:status).to_i != 0
    py_all_failed || rb_all_failed
  end

  def write_report(report)
    timestamp = Time.now.utc.strftime("%Y%m%dT%H%M%SZ")
    latest_path = RESULTS_DIR.join("latest.json")
    timestamped_path = RESULTS_DIR.join("benchmark_#{timestamp}.json")

    payload = JSON.pretty_generate(report)
    latest_path.write("#{payload}\n")
    timestamped_path.write("#{payload}\n")
  end

  def ensure_command_success!(command, label:, timeout:)
    result = run_capture(command, chdir: ROOT, env: {}, timeout: timeout)
    return if result.fetch(:status).zero?

    warn "Command failed during #{label}: #{command.join(' ')}"
    warn "STDOUT:\n#{result.fetch(:stdout)}"
    warn "STDERR:\n#{result.fetch(:stderr)}"
    abort "Aborting benchmark setup"
  end

  def capture_stdout(command)
    result = run_capture(command, chdir: ROOT, env: {}, timeout: 60)
    unless result.fetch(:status).zero?
      raise "Command failed: #{command.join(' ')}\n#{result.fetch(:stderr)}"
    end

    result.fetch(:stdout)
  end

  def run_capture(command, chdir:, env:, timeout:)
    stdout = +""
    stderr = +""
    timed_out = false
    status = nil

    started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    elapsed = nil

    Open3.popen3(env, *command, chdir: chdir.to_s) do |stdin, out, err, wait_thr|
      stdin.close
      out_reader = Thread.new { out.read }
      err_reader = Thread.new { err.read }

      if timeout && timeout > 0
        if wait_thr.join(timeout).nil?
          timed_out = true
          begin
            Process.kill("TERM", wait_thr.pid)
          rescue Errno::ESRCH
            nil
          end
          wait_thr.join(5)
          if wait_thr.alive?
            begin
              Process.kill("KILL", wait_thr.pid)
            rescue Errno::ESRCH
              nil
            end
            wait_thr.join
          end
        end
      else
        wait_thr.join
      end

      stdout = out_reader.value
      stderr = err_reader.value
      status = wait_thr.value
    end

    elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at

    status_code = timed_out ? 124 : (status&.exitstatus || (status&.success? ? 0 : 1))

    {
      status: status_code,
      stdout: stdout,
      stderr: stderr,
      elapsed: elapsed,
      timeout: timed_out
    }
  rescue StandardError => e
    elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at
    {
      status: 1,
      stdout: stdout,
      stderr: "#{e.class}: #{e.message}\n#{Array(e.backtrace).join("\n")}",
      elapsed: elapsed,
      timeout: false
    }
  end

  def write_log(log_file:, command:, result:)
    content = +""
    content << "command: #{command.join(' ')}\n"
    content << "status: #{result.fetch(:status)}\n"
    content << "elapsed_seconds: #{format('%.6f', result.fetch(:elapsed))}\n"
    content << "timed_out: #{result.fetch(:timeout)}\n"
    content << "\nSTDOUT:\n#{result.fetch(:stdout)}\n"
    content << "\nSTDERR:\n#{result.fetch(:stderr)}\n"
    log_file.write(content)
  end
end

mode = ARGV.shift || "dsl"
BenchmarkRunner.run(mode: mode)
