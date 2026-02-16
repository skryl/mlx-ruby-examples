# frozen_string_literal: true

require "rbconfig"
require "rake/testtask"
require "shellwords"

ROOT = File.expand_path(__dir__)
TEST_SCRIPTS = Dir.glob(File.join(ROOT, "**", "test.rb")).sort.freeze
REQUIREMENTS_PATH = File.join(ROOT, "requirements.txt").freeze

def python_cmd
  cmd = Shellwords.split(ENV.fetch("PYTHON_BIN", "python3"))
  raise ArgumentError, "PYTHON_BIN must not be empty" if cmd.empty?

  cmd
end

def ensure_python_requirements
  return if ENV["SKIP_PYTHON_REQUIREMENTS"] == "1"
  return unless File.exist?(REQUIREMENTS_PATH)

  install_cmd = python_cmd + ["-m", "pip", "install", "-r", REQUIREMENTS_PATH]
  puts "==> Installing Python requirements from #{REQUIREMENTS_PATH}"
  success = system(*install_cmd)
  return if success

  abort "Failed to install Python requirements using: #{install_cmd.join(' ')}"
end

def run_benchmark_task(mode)
  ruby = RbConfig.ruby
  runner = File.join(ROOT, "benchmark", "runner.rb")
  success = system(ruby, runner, mode)
  return if success

  abort(mode == "no_dsl" ? "benchmark:no_dsl failed" : "benchmark failed")
end

namespace :test do
  desc "Install Python dependencies for model tests"
  task :deps do
    ensure_python_requirements
  end

  desc "Run legacy model test scripts"
  task :old do
    ruby = RbConfig.ruby
    failures = []

    TEST_SCRIPTS.each do |script|
      relative = script.delete_prefix("#{ROOT}/")
      puts "==> #{relative}"
      success = system(ruby, script)
      failures << relative unless success
    end

    next if failures.empty?

    warn
    warn "Failed test scripts:"
    failures.each { |script| warn "  - #{script}" }
    abort "test:old failed"
  end
end

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
end

desc "Benchmark Ruby DSL models against Python mlx-examples equivalents"
task :benchmark do
  run_benchmark_task("dsl")
end

namespace :benchmark do
  desc "Benchmark Ruby no_dsl models against Python mlx-examples equivalents"
  task :no_dsl do
    run_benchmark_task("no_dsl")
  end
end

task default: :test
