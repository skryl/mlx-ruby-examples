# frozen_string_literal: true

require_relative "test_helper"

class LoraModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("lora/test.rb")
  end
end
