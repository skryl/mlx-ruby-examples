# frozen_string_literal: true

require_relative "test_helper"

class GcnModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("gcn/test.rb")
  end
end
