# frozen_string_literal: true

script = ARGV.shift
abort "Usage: ruby benchmark/ruby/run_with_dryrun.rb path/to/test.rb" if script.nil? || script.empty?

script_path = File.expand_path(script, Dir.pwd)
abort "Missing script: #{script_path}" unless File.exist?(script_path)

def run_script(script_path, dryrun:)
  previous_progname = $PROGRAM_NAME
  previous_argv = ARGV.dup
  previous_dryrun = ENV["MLX_BENCHMARK_DRYRUN"]

  ENV["MLX_BENCHMARK_DRYRUN"] = dryrun ? "1" : "0"
  ARGV.replace([])
  $PROGRAM_NAME = script_path

  load script_path
rescue SystemExit => e
  status = e.status.to_i
  raise if status != 0
ensure
  $PROGRAM_NAME = previous_progname
  ARGV.replace(previous_argv)
  if previous_dryrun.nil?
    ENV.delete("MLX_BENCHMARK_DRYRUN")
  else
    ENV["MLX_BENCHMARK_DRYRUN"] = previous_dryrun
  end
end

run_script(script_path, dryrun: true)
run_script(script_path, dryrun: false)
