# frozen_string_literal: true

require_relative "test_helper"

class BertModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("bert/test.rb")
  end
end
