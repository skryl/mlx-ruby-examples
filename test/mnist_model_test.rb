# frozen_string_literal: true

require_relative "test_helper"

class MnistModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("mnist/test.rb")
  end
end
