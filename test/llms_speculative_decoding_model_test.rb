# frozen_string_literal: true

require_relative "test_helper"

class LlmsSpeculativeDecodingModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("llms/speculative_decoding/test.rb")
  end
end
