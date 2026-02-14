# frozen_string_literal: true

require "time"

require_relative "model_io"
require_relative "sampler"

module StableDiffusionExample
  class StableDiffusion
    attr_reader :dtype, :diffusion_config, :unet, :text_encoder, :autoencoder, :sampler, :tokenizer

    def initialize(model: DEFAULT_MODEL, float16: false)
      @dtype = float16 ? MLX::Core.float16 : MLX::Core.float32
      @diffusion_config = StableDiffusionExample.load_diffusion_config(model)
      @unet = StableDiffusionExample.load_unet(model, float16)
      @text_encoder = StableDiffusionExample.load_text_encoder(model, float16)
      @autoencoder = StableDiffusionExample.load_autoencoder(model, false)
      @sampler = SimpleEulerSampler.new(diffusion_config)
      @tokenizer = StableDiffusionExample.load_tokenizer(model)
    end

    def ensure_models_are_loaded
      MLX::Core.eval(unet.parameters)
      MLX::Core.eval(text_encoder.parameters)
      MLX::Core.eval(autoencoder.parameters)
    end

    def generate_latents(
      text,
      n_images: 1,
      num_steps: 50,
      cfg_weight: 7.5,
      negative_text: "",
      latent_size: [64, 64],
      seed: nil
    )
      seed = Time.now.to_i if seed.nil?
      MLX::Core.random_seed(seed.to_i)

      conditioning = get_text_conditioning(
        text,
        n_images: n_images,
        cfg_weight: cfg_weight,
        negative_text: negative_text
      )

      x_t = sampler.sample_prior(
        [n_images, latent_size[0], latent_size[1], autoencoder.latent_channels],
        dtype: dtype
      )

      denoising_loop(
        x_t,
        sampler.max_time,
        conditioning,
        num_steps: num_steps,
        cfg_weight: cfg_weight
      )
    end

    def generate_latents_from_image(
      image,
      text,
      n_images: 1,
      strength: 0.8,
      num_steps: 50,
      cfg_weight: 7.5,
      negative_text: "",
      seed: nil
    )
      seed = Time.now.to_i if seed.nil?
      MLX::Core.random_seed(seed.to_i)

      start_step = sampler.max_time * strength
      steps = [(num_steps * strength).to_i, 1].max

      conditioning = get_text_conditioning(
        text,
        n_images: n_images,
        cfg_weight: cfg_weight,
        negative_text: negative_text
      )

      x0, _ = autoencoder.encode(MLX::Core.expand_dims(image, 0))
      x0 = repeat_axis0(x0, n_images)
      x_t = sampler.add_noise(x0, MLX::Core.array(start_step, dtype))

      denoising_loop(
        x_t,
        start_step,
        conditioning,
        num_steps: steps,
        cfg_weight: cfg_weight
      )
    end

    def decode(x_t)
      x = autoencoder.decode(x_t)
      x = MLX::Core.add(MLX::Core.divide(x, 2.0), 0.5)
      MLX::Core.clip(x, 0.0, 1.0)
    end

    protected

    def tokenize(tokenizer, text, negative_text: nil)
      tokens = [tokenizer.tokenize(text)]
      tokens << tokenizer.tokenize(negative_text) unless negative_text.nil?
      MLX::Core.array(tokens, MLX::Core.int32)
    end

    def get_text_conditioning(text, n_images:, cfg_weight:, negative_text: "")
      tokens = tokenize(tokenizer, text, negative_text: (cfg_weight > 1 ? negative_text : nil))
      conditioning = text_encoder.call(tokens).last_hidden_state
      conditioning = repeat_axis0(conditioning, n_images) if n_images > 1
      conditioning
    end

    def denoising_step(x_t, t, t_prev, conditioning, cfg_weight:, text_time: nil)
      if cfg_weight > 1
        x_t_unet = MLX::Core.concatenate([x_t, x_t], 0)
      else
        x_t_unet = x_t
      end

      t_unet = MLX::Core.broadcast_to(t, [x_t_unet.shape[0]])
      eps_pred = unet.call(x_t_unet, t_unet, encoder_x: conditioning, text_time: text_time)

      if cfg_weight > 1
        n = eps_pred.shape[0] / 2
        eps_text = MLX::Core.slice(
          eps_pred,
          [0, 0, 0, 0],
          [n, eps_pred.shape[1], eps_pred.shape[2], eps_pred.shape[3]]
        )
        eps_neg = MLX::Core.slice(
          eps_pred,
          [n, 0, 0, 0],
          [2 * n, eps_pred.shape[1], eps_pred.shape[2], eps_pred.shape[3]]
        )
        diff = MLX::Core.subtract(eps_text, eps_neg)
        eps_pred = MLX::Core.add(eps_neg, MLX::Core.multiply(cfg_weight, diff))
      end

      sampler.step(eps_pred, x_t, t, t_prev)
    end

    def denoising_loop(x_t, start_time, conditioning, num_steps:, cfg_weight:, text_time: nil)
      Enumerator.new do |yielder|
        sampler.timesteps(num_steps, start_time: start_time, dtype: dtype).each do |t, t_prev|
          x_t = denoising_step(
            x_t,
            t,
            t_prev,
            conditioning,
            cfg_weight: cfg_weight,
            text_time: text_time
          )
          yielder << x_t
        end
      end
    end

    def repeat_axis0(x, repeats)
      return x if repeats <= 1

      arr = x.to_a
      repeated = arr.flat_map { |row| Array.new(repeats) { row } }
      MLX::Core.array(repeated, x.dtype)
    end
  end

  class StableDiffusionXL < StableDiffusion
    attr_reader :text_encoder_1, :tokenizer_1, :text_encoder_2, :tokenizer_2

    def initialize(model: "stabilityai/sdxl-turbo", float16: false)
      super(model: model, float16: float16)

      @sampler = SimpleEulerAncestralSampler.new(diffusion_config)

      @text_encoder_1 = text_encoder
      @tokenizer_1 = tokenizer

      @text_encoder_2 = StableDiffusionExample.load_text_encoder(
        model,
        float16,
        model_key: "text_encoder_2",
        config_key: "text_encoder_2_config"
      )
      @tokenizer_2 = StableDiffusionExample.load_tokenizer(
        model,
        vocab_key: "tokenizer_2_vocab",
        merges_key: "tokenizer_2_merges"
      )
    end

    def ensure_models_are_loaded
      MLX::Core.eval(unet.parameters)
      MLX::Core.eval(text_encoder_1.parameters)
      MLX::Core.eval(text_encoder_2.parameters)
      MLX::Core.eval(autoencoder.parameters)
    end

    def generate_latents(
      text,
      n_images: 1,
      num_steps: 2,
      cfg_weight: 0.0,
      negative_text: "",
      latent_size: [64, 64],
      seed: nil
    )
      seed = Time.now.to_i if seed.nil?
      MLX::Core.random_seed(seed.to_i)

      conditioning, pooled = get_text_conditioning_xl(
        text,
        n_images: n_images,
        cfg_weight: cfg_weight,
        negative_text: negative_text
      )

      text_time = [
        pooled,
        MLX::Core.array(Array.new(pooled.shape[0]) { [512, 512, 0, 0, 512, 512.0] }, MLX::Core.float32)
      ]

      x_t = sampler.sample_prior(
        [n_images, latent_size[0], latent_size[1], autoencoder.latent_channels],
        dtype: dtype
      )

      denoising_loop(
        x_t,
        sampler.max_time,
        conditioning,
        num_steps: num_steps,
        cfg_weight: cfg_weight,
        text_time: text_time
      )
    end

    def generate_latents_from_image(
      image,
      text,
      n_images: 1,
      strength: 0.8,
      num_steps: 2,
      cfg_weight: 0.0,
      negative_text: "",
      seed: nil
    )
      seed = Time.now.to_i if seed.nil?
      MLX::Core.random_seed(seed.to_i)

      start_step = sampler.max_time * strength
      steps = [(num_steps * strength).to_i, 1].max

      conditioning, pooled = get_text_conditioning_xl(
        text,
        n_images: n_images,
        cfg_weight: cfg_weight,
        negative_text: negative_text
      )
      text_time = [
        pooled,
        MLX::Core.array(Array.new(pooled.shape[0]) { [512, 512, 0, 0, 512, 512.0] }, MLX::Core.float32)
      ]

      x0, _ = autoencoder.encode(MLX::Core.expand_dims(image, 0))
      x0 = repeat_axis0(x0, n_images)
      x_t = sampler.add_noise(x0, MLX::Core.array(start_step, dtype))

      denoising_loop(
        x_t,
        start_step,
        conditioning,
        num_steps: steps,
        cfg_weight: cfg_weight,
        text_time: text_time
      )
    end

    private

    def get_text_conditioning_xl(text, n_images:, cfg_weight:, negative_text: "")
      tokens_1 = tokenize(tokenizer_1, text, negative_text: (cfg_weight > 1 ? negative_text : nil))
      tokens_2 = tokenize(tokenizer_2, text, negative_text: (cfg_weight > 1 ? negative_text : nil))

      out1 = text_encoder_1.call(tokens_1)
      out2 = text_encoder_2.call(tokens_2)

      hs1 = out1.hidden_states[-1]
      hs2 = out2.hidden_states[-1]
      conditioning = MLX::Core.concatenate([hs1, hs2], -1)
      pooled = out2.pooled_output

      if n_images > 1
        conditioning = repeat_axis0(conditioning, n_images)
        pooled = repeat_axis0(pooled, n_images)
      end

      [conditioning, pooled]
    end
  end
end
