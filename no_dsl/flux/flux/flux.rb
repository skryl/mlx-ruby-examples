# frozen_string_literal: true

require_relative "lora"
require_relative "sampler"
require_relative "utils"

module FluxExample
  class FluxPipeline
    attr_reader :dtype, :name, :t5_padding, :ae, :flow, :clip, :clip_tokenizer, :t5, :t5_tokenizer, :sampler

    def initialize(name, t5_padding: true, hf_download: true, python_bin: ENV.fetch("PYTHON_BIN", "python3"))
      @dtype = MLX::Core.bfloat16
      @name = FluxExample.resolve_model_name(name)
      @t5_padding = t5_padding

      @ae = FluxExample.load_ae(@name, hf_download_enabled: hf_download, python_bin: python_bin)
      @flow = FluxExample.load_flow_model(@name, hf_download_enabled: hf_download, python_bin: python_bin)
      @clip = FluxExample.load_clip(@name)
      @clip_tokenizer = FluxExample.load_clip_tokenizer(@name)
      @t5 = FluxExample.load_t5(@name)
      @t5_tokenizer = FluxExample.load_t5_tokenizer(@name)
      @sampler = FluxSampler.new(@name)
    end

    def ensure_models_are_loaded
      MLX::Core.eval(ae.parameters, flow.parameters, clip.parameters, t5.parameters)
    end

    def reload_text_encoders
      @t5 = FluxExample.load_t5(name)
      @clip = FluxExample.load_clip(name)
    end

    def tokenize(text)
      t5_tokens = t5_tokenizer.encode(text, pad: t5_padding)
      clip_tokens = clip_tokenizer.encode(text)
      [t5_tokens, clip_tokens]
    end

    def prepare_latent_images(x)
      b, h, w, c = x.shape
      x = MLX::Core.reshape(x, [b, h / 2, 2, w / 2, 2, c])
      x = MLX::Core.transpose(x, [0, 1, 3, 5, 2, 4])
      x = MLX::Core.reshape(x, [b, h * w / 4, c * 4])

      i = Array.new(h / 2) { Array.new(w / 2, 0) }
      x_ids = []
      (0...(h / 2)).each do |jj|
        (0...(w / 2)).each do |kk|
          x_ids << [0, jj, kk]
        end
      end
      x_ids = Array.new(b) { x_ids }
      x_ids = MLX::Core.array(x_ids, MLX::Core.int32)

      [x, x_ids]
    end

    def prepare_conditioning(n_images, t5_tokens, clip_tokens)
      txt = t5.call(t5_tokens)
      if txt.shape[0] == 1 && n_images > 1
        txt = repeat_axis0(txt, n_images)
      end
      txt_ids = MLX::Core.zeros([n_images, txt.shape[1], 3], MLX::Core.int32)

      vec = clip.call(clip_tokens).pooled_output
      if vec.shape[0] == 1 && n_images > 1
        vec = repeat_axis0(vec, n_images)
      end

      [txt, txt_ids, vec]
    end

    def denoising_loop(
      x_t,
      x_ids,
      txt,
      txt_ids,
      vec,
      num_steps: 35,
      guidance: 4.0,
      start: 1.0,
      stop: 0.0
    )
      b = x_t.shape[0]
      scalar = lambda { |xv| MLX::Core.full([b], xv, dtype) }
      guidance_vec = scalar.call(guidance)

      timesteps = sampler.timesteps(num_steps, x_t.shape[1], start: start, stop: stop)
      Enumerator.new do |yielder|
        num_steps.times do |i|
          t = timesteps[i]
          t_prev = timesteps[i + 1]

          pred = flow.call(
            img: x_t,
            img_ids: x_ids,
            txt: txt,
            txt_ids: txt_ids,
            y: vec,
            timesteps: scalar.call(t),
            guidance: guidance_vec
          )
          x_t = sampler.step(pred, x_t, t, t_prev)
          yielder << x_t
        end
      end
    end

    def generate_latents(
      text,
      n_images: 1,
      num_steps: 35,
      guidance: 4.0,
      latent_size: [64, 64],
      seed: nil
    )
      MLX::Core.random_seed(seed.to_i) unless seed.nil?

      x_t = sampler.sample_prior([n_images, latent_size[0], latent_size[1], 16], dtype: dtype)
      x_t, x_ids = prepare_latent_images(x_t)

      t5_tokens, clip_tokens = tokenize(text)
      txt, txt_ids, vec = prepare_conditioning(n_images, t5_tokens, clip_tokens)

      Enumerator.new do |yielder|
        yielder << [x_t, x_ids, txt, txt_ids, vec]
        denoising_loop(
          x_t,
          x_ids,
          txt,
          txt_ids,
          vec,
          num_steps: num_steps,
          guidance: guidance
        ).each { |step_latent| yielder << step_latent }
      end
    end

    def decode(x, latent_size = [64, 64])
      h, w = latent_size
      x = MLX::Core.reshape(x, [x.shape[0], h / 2, w / 2, x.shape[2] / 4, 2, 2])
      x = MLX::Core.transpose(x, [0, 1, 4, 2, 5, 3])
      x = MLX::Core.reshape(x, [x.shape[0], h, w, x.shape[5]])
      x = ae.decode(x)
      MLX::Core.multiply(0.5, MLX::Core.clip(MLX::Core.add(x, 1.0), 0.0, 2.0))
    end

    def generate_images(
      text,
      n_images: 1,
      num_steps: 35,
      guidance: 4.0,
      latent_size: [64, 64],
      seed: nil,
      reload_text_encoders: true,
      progress: true
    )
      _ = progress
      latents = generate_latents(
        text,
        n_images: n_images,
        num_steps: num_steps,
        guidance: guidance,
        latent_size: latent_size,
        seed: seed
      )
      MLX::Core.eval(*latents.next)

      self.reload_text_encoders if reload_text_encoders

      x_t = nil
      latents.each do |latent|
        x_t = latent
        MLX::Core.eval(x_t)
      end

      images = []
      x_t.shape[0].times do |i|
        img = decode(
          MLX::Core.slice(x_t, [i, 0, 0], [i + 1, x_t.shape[1], x_t.shape[2]]),
          latent_size
        )
        MLX::Core.eval(img)
        images << img
      end
      out = MLX::Core.concatenate(images, 0)
      MLX::Core.eval(out)
      out
    end

    def training_loss(x0, t5_features, clip_features, guidance)
      txt = t5_features
      txt_ids = MLX::Core.zeros([txt.shape[0], txt.shape[1], 3], MLX::Core.int32)
      vec = clip_features

      x0, x_ids = prepare_latent_images(x0)

      t = sampler.random_timesteps(x0.shape[0], x0.shape[1], dtype: dtype)
      eps = MLX::Core.normal(x0.shape).astype(dtype)
      x_t = sampler.add_noise(x0, t, noise: eps)
      x_t = MLX::Core.stop_gradient(x_t)

      pred = flow.call(
        img: x_t,
        img_ids: x_ids,
        txt: txt,
        txt_ids: txt_ids,
        y: vec,
        timesteps: t,
        guidance: guidance
      )

      diff = MLX::Core.add(MLX::Core.add(pred, x0), MLX::Core.multiply(-1.0, eps))
      MLX::Core.mean(MLX::Core.square(diff))
    end

    def linear_to_lora_layers(rank: 8, num_blocks: -1)
      all_blocks = (flow.double_blocks + flow.single_blocks).reverse
      nb = (num_blocks.positive? ? num_blocks : all_blocks.length)
      all_blocks.first(nb).each do |block|
        next unless block.respond_to?(:named_modules)

        loras = []
        block.named_modules.each do |module_name, module_obj|
          if module_obj.is_a?(MLX::NN::Linear)
            loras << [module_name, LoRALinear.from_base(module_obj, r: rank)]
          end
        end
        next if loras.empty?

        begin
          block.update_modules(MLX::Utils.tree_unflatten(loras))
        rescue StandardError
          # Best effort: some module trees are not directly swappable in-place.
        end
      end
    end

    def fuse_lora_layers
      fused = []
      return unless flow.respond_to?(:named_modules)

      flow.named_modules.each do |module_name, module_obj|
        fused << [module_name, module_obj.fuse] if module_obj.is_a?(LoRALinear) && module_obj.respond_to?(:fuse)
      end
      return if fused.empty?

      begin
        flow.update_modules(MLX::Utils.tree_unflatten(fused))
      rescue StandardError
        # Best effort fuse; ignore unsupported module-path shapes.
      end
    end

    private

    def repeat_axis0(x, repeats)
      return x if repeats <= 1

      arr = x.to_a
      out = arr.flat_map { |row| Array.new(repeats) { row } }
      MLX::Core.array(out, x.dtype)
    end
  end
end
