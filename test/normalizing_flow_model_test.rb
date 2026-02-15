# frozen_string_literal: true

require_relative "test_helper"

class NormalizingFlowModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("normalizing_flow/test.rb")
  end
end
