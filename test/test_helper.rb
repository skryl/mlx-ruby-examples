# frozen_string_literal: true

require "minitest/autorun"
require "open3"
require "rbconfig"

module ModelScriptTestHelper
  private

  def assert_model_script_passes(script_relative_path)
    script_path = File.expand_path(File.join(__dir__, "..", script_relative_path))
    stdout, stderr, status = Open3.capture3(RbConfig.ruby, script_path)
    message = +"Expected #{script_relative_path} to pass.\n"
    message << "Exit status: #{status.exitstatus}\n"
    message << "STDOUT:\n#{stdout}\n"
    message << "STDERR:\n#{stderr}\n"
    assert status.success?, message
  end
end
