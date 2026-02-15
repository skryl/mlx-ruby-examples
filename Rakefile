# frozen_string_literal: true

require "rbconfig"
require "rake/testtask"

ROOT = File.expand_path(__dir__)
TEST_SCRIPTS = Dir.glob(File.join(ROOT, "**", "test.rb")).sort.freeze

namespace :test do
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

task default: :test
