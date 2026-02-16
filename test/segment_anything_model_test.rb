# frozen_string_literal: true

require_relative "test_helper"

class SegmentAnythingModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("segment_anything/test.rb")
  end
end
