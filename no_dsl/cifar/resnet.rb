# frozen_string_literal: true

require "pathname"

dsl_lib = File.join(File.expand_path("..", __dir__), "codex-dsl", "lib")
$LOAD_PATH.unshift(dsl_lib) unless $LOAD_PATH.include?(dsl_lib)

require "mlx"

module CifarExample
  class ShortcutA < MLX::NN::Module
    def initialize(dims)
      super()
      @dims = dims
    end

    def call(x)
      h_indices = MLX::Core.array((0...x.shape[1]).step(2).to_a, MLX::Core.int32)
      w_indices = MLX::Core.array((0...x.shape[2]).step(2).to_a, MLX::Core.int32)
      out = MLX::Core.take(x, h_indices, 1)
      out = MLX::Core.take(out, w_indices, 2)
      pad = @dims / 4
      MLX::Core.pad(out, [[0, 0], [0, 0], [0, 0], [pad, pad]])
    end
  end

  class Block < MLX::NN::Module
    def initialize(in_dims, dims, stride: 1)
      super()
      self.conv1 = MLX::NN::Conv2d.new(
        in_dims,
        dims,
        3,
        stride: stride,
        padding: 1,
        bias: false
      )
      self.bn1 = MLX::NN::BatchNorm.new(dims)

      self.conv2 = MLX::NN::Conv2d.new(
        dims,
        dims,
        3,
        stride: 1,
        padding: 1,
        bias: false
      )
      self.bn2 = MLX::NN::BatchNorm.new(dims)
      self.shortcut = stride == 1 ? nil : ShortcutA.new(dims)
    end

    def call(x)
      out = MLX::NN.relu(bn1.call(conv1.call(x)))
      out = bn2.call(conv2.call(out))
      out = if shortcut.nil?
        MLX::Core.add(out, x)
      else
        MLX::Core.add(out, shortcut.call(x))
      end
      MLX::NN.relu(out)
    end
  end

  class ResNet < MLX::NN::Module
    def initialize(block_class, num_blocks, num_classes: 10)
      super()
      self.conv1 = MLX::NN::Conv2d.new(3, 16, 3, stride: 1, padding: 1, bias: false)
      self.bn1 = MLX::NN::BatchNorm.new(16)

      self.layer1 = make_layer(block_class, 16, 16, num_blocks[0], stride: 1)
      self.layer2 = make_layer(block_class, 16, 32, num_blocks[1], stride: 2)
      self.layer3 = make_layer(block_class, 32, 64, num_blocks[2], stride: 2)
      self.linear = MLX::NN::Linear.new(64, num_classes)
    end

    def num_params
      MLX::Utils.tree_flatten(parameters).sum { |_k, x| x.size }
    end

    def call(x)
      x = MLX::NN.relu(bn1.call(conv1.call(x)))
      x = layer1.call(x)
      x = layer2.call(x)
      x = layer3.call(x)
      x = MLX::Core.mean(x, 1)
      x = MLX::Core.mean(x, 1)
      linear.call(x)
    end

    private

    def make_layer(block_class, in_dims, dims, count, stride:)
      strides = [stride] + Array.new(count - 1, 1)
      layers = []
      strides.each do |s|
        layers << block_class.new(in_dims, dims, stride: s)
        in_dims = dims
      end
      MLX::NN::Sequential.new(*layers)
    end
  end

  module_function

  def resnet20(**kwargs)
    ResNet.new(Block, [3, 3, 3], **kwargs)
  end

  def resnet32(**kwargs)
    ResNet.new(Block, [5, 5, 5], **kwargs)
  end

  def resnet44(**kwargs)
    ResNet.new(Block, [7, 7, 7], **kwargs)
  end

  def resnet56(**kwargs)
    ResNet.new(Block, [9, 9, 9], **kwargs)
  end

  def resnet110(**kwargs)
    ResNet.new(Block, [18, 18, 18], **kwargs)
  end

  def resnet1202(**kwargs)
    ResNet.new(Block, [200, 200, 200], **kwargs)
  end
end
