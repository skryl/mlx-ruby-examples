# frozen_string_literal: true

require_relative "test_helper"

class LlmsGgufLlmModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("llms/gguf_llm/test.rb")
  end
end
