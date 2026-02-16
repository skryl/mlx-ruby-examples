# frozen_string_literal: true

require_relative "test_helper"

class TransformerLmModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("transformer_lm/test.rb")
  end
end
