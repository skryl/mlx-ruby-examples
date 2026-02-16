# frozen_string_literal: true

require "minitest/autorun"
require "open3"
require "rbconfig"

module ModelScriptTestHelper
  private

  def assert_model_script_passes(script_relative_path)
    assert_single_model_script_passes(script_relative_path)

    no_dsl_path = File.join("no_dsl", script_relative_path)
    absolute_no_dsl_path = File.expand_path(File.join(__dir__, "..", no_dsl_path))
    if File.exist?(absolute_no_dsl_path)
      assert_single_model_script_passes(no_dsl_path)
    end
  end

  def assert_single_model_script_passes(script_relative_path)
    script_path = File.expand_path(File.join(__dir__, "..", script_relative_path))
    script_dir = File.dirname(script_path)
    script_name = File.basename(script_path)
    stdout, stderr, status = Open3.capture3(RbConfig.ruby, script_name, chdir: script_dir)
    message = +"Expected #{script_relative_path} to pass.\n"
    message << "Exit status: #{status.exitstatus}\n"
    message << "STDOUT:\n#{stdout}\n"
    message << "STDERR:\n#{stderr}\n"
    assert status.success?, message
  end
end
