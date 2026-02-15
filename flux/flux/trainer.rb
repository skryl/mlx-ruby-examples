# frozen_string_literal: true


require "mlx"

require_relative "datasets"
require_relative "flux"

module FluxExample
  class Trainer
    attr_reader :flux, :dataset, :args, :latents, :t5_features, :clip_features

    def initialize(flux, dataset, args)
      @flux = flux
      @dataset = dataset
      @args = args
      @latents = []
      @t5_features = []
      @clip_features = []
    end

    def random_crop_resize(img)
      resolution = args.resolution
      x = img.respond_to?(:shape) ? img : MLX::Core.array(img)
      h = x.shape[0]
      w = x.shape[1]

      target_h = [resolution[1], h].min
      target_w = [resolution[0], w].min

      y0 = ((h - target_h) / 2.0).floor
      x0 = ((w - target_w) / 2.0).floor
      cropped = MLX::Core.slice(x, [y0, x0, 0], [y0 + target_h, x0 + target_w, x.shape[2]])

      resize_nearest(cropped, resolution[1], resolution[0])
    end

    def encode_image(input_img, num_augmentations)
      num_augmentations.times do
        img = random_crop_resize(input_img)
        img = MLX::Core.multiply(2.0, img.astype(flux.dtype))
        img = MLX::Core.subtract(img, 1.0)
        x0 = flux.ae.encode(MLX::Core.expand_dims(img, 0))
        x0 = x0.astype(flux.dtype)
        MLX::Core.eval(x0)
        latents << x0
      end
    end

    def encode_prompt(prompt)
      t5_tok, clip_tok = flux.tokenize([prompt])
      t5_feat = flux.t5.call(t5_tok)
      clip_feat = flux.clip.call(clip_tok).pooled_output
      MLX::Core.eval(t5_feat, clip_feat)
      t5_features << t5_feat
      clip_features << clip_feat
    end

    def encode_dataset
      dataset.each do |image, prompt|
        encode_image(image, args.num_augmentations)
        encode_prompt(prompt)
      end
    end

    def iterate(batch_size)
      xs = MLX::Core.concatenate(latents, 0)
      t5 = MLX::Core.concatenate(t5_features, 0)
      clip = MLX::Core.concatenate(clip_features, 0)
      MLX::Core.eval(xs, t5, clip)

      n_aug = args.num_augmentations
      Enumerator.new do |yielder|
        loop do
          perm = rand_perm(latents.length)
          cond = perm.map { |v| v / n_aug }
          perm.each_slice(batch_size) do |slice|
            c_slice = slice.map { |i| cond[i] }
            x_i = MLX::Core.array(slice, MLX::Core.int32)
            c_i = MLX::Core.array(c_slice, MLX::Core.int32)
            yielder << [MLX::Core.take(xs, x_i, 0), MLX::Core.take(t5, c_i, 0), MLX::Core.take(clip, c_i, 0)]
          end
        end
      end
    end

    private

    def rand_perm(n)
      arr = (0...n).to_a
      (n - 1).downto(1) do |i|
        j = rand(i + 1)
        arr[i], arr[j] = arr[j], arr[i]
      end
      arr
    end

    def resize_nearest(image, out_h, out_w)
      in_h = image.shape[0]
      in_w = image.shape[1]
      h_idx = Array.new(out_h) { |i| [(i.to_f * in_h / out_h).floor, in_h - 1].min }
      w_idx = Array.new(out_w) { |i| [(i.to_f * in_w / out_w).floor, in_w - 1].min }
      h_idx = MLX::Core.array(h_idx, MLX::Core.int32)
      w_idx = MLX::Core.array(w_idx, MLX::Core.int32)
      out = MLX::Core.take(image, h_idx, 0)
      MLX::Core.take(out, w_idx, 1)
    end
  end
end
