# frozen_string_literal: true

require "optparse"

require_relative "image_processor"
require_relative "model"

module ClipExample
  class LinearProbeHead < MLX::DSL::Model
    option :in_dim
    option :out_dim

    layer :classifier, MLX::NN::Linear, -> { in_dim }, -> { out_dim }

    def call(x)
      classifier.call(x)
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = {
    samples: 256,
    classes: 5,
    image_size: 64,
    batch_size: 32,
    epochs: 5,
    lr: 1e-2,
    seed: 0
  }

  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby clip/linear_probe.rb [options]"
    opts.on("--samples N", Integer, "Number of synthetic samples") { |v| options[:samples] = v }
    opts.on("--classes N", Integer, "Number of classes") { |v| options[:classes] = v }
    opts.on("--image-size N", Integer, "Image size") { |v| options[:image_size] = v }
    opts.on("--batch-size N", Integer, "Batch size") { |v| options[:batch_size] = v }
    opts.on("--epochs N", Integer, "Epochs") { |v| options[:epochs] = v }
    opts.on("--lr N", Float, "Learning rate") { |v| options[:lr] = v }
    opts.on("--seed N", Integer, "PRNG seed") { |v| options[:seed] = v }
  end
  parser.parse!

  MLX::Core.random_seed(options[:seed])
  processor = ClipExample::ImageProcessor.new(image_size: options[:image_size])
  clip_model = ClipExample::CLIPModel.new(
    vocab_size: 259,
    embed_dim: 64,
    text_width: 128,
    vision_width: 128,
    image_size: options[:image_size],
    patch_size: 8
  )
  linear = ClipExample::LinearProbeHead.new(in_dim: 64, out_dim: options[:classes])
  optimizer = MLX::Optimizers::Adam.new(learning_rate: options[:lr])

  labels = MLX::Core.array(
    Array.new(options[:samples]) { rand(options[:classes]) },
    MLX::Core.int32
  )
  prototypes = MLX::Core.random_uniform(
    [options[:classes], options[:image_size], options[:image_size], 3],
    0.0,
    255.0,
    MLX::Core.float32
  )
  images = MLX::Core.take(prototypes, labels, 0)
  images = MLX::Core.add(
    images,
    MLX::Core.random_uniform(images.shape, -10.0, 10.0, MLX::Core.float32)
  )
  images = processor.call(images)
  features = clip_model.encode_image(images)
  MLX::Core.eval(features)

  trainer = linear.trainer(optimizer: optimizer) do |x:, y:|
    logits = linear.call(x)
    MLX::Core.mean(MLX::NN::Losses.cross_entropy(logits, y))
  end

  trainer.after_epoch do |ctx|
    puts format("Epoch %d: loss %.4f", ctx.fetch(:epoch), ctx.fetch(:epoch_loss).to_f)
  end

  train_data = lambda do |epoch:, **_kwargs|
    MLX::DSL::Data
      .from(0...options[:samples])
      .shuffle(seed: options[:seed] + epoch.to_i)
      .batch(options[:batch_size])
      .map do |batch_ids|
        ids = MLX::Core.array(batch_ids, MLX::Core.int32)
        x = MLX::Core.take(features, ids, 0)
        y = MLX::Core.take(labels, ids, 0)
        [x, y]
      end
  end

  trainer.fit_report(
    train_data,
    epochs: options[:epochs],
    collate: :xy,
    reduce: :mean,
    keep_losses: false,
    strict_data_reuse: true
  )

  logits = linear.call(features)
  preds = MLX::Core.argmax(logits, 1)
  acc = MLX::Core.mean(MLX::Core.equal(preds, labels))
  MLX::Core.eval(acc)
  puts format("Linear probe accuracy: %.3f", acc.item.to_f)
end
