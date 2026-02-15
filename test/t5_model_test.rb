# frozen_string_literal: true

require_relative "test_helper"

class T5ModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("t5/test.rb")
  end
end
