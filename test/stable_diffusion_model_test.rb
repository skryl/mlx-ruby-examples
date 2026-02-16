# frozen_string_literal: true

require_relative "test_helper"

class StableDiffusionModelTest < Minitest::Test
  include ModelScriptTestHelper

  def test_model_script
    assert_model_script_passes("stable_diffusion/test.rb")
  end
end
