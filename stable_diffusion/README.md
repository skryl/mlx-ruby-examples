# Stable Diffusion (MLX Ruby + DSL)

Ruby port of `mlx-examples/stable_diffusion` with a lightweight synthetic-first pipeline.

## Files

- `stable_diffusion/config.rb`: model config objects.
- `stable_diffusion/tokenizer.rb`: tokenizer utility.
- `stable_diffusion/clip.rb`: CLIP-like text encoder.
- `stable_diffusion/unet.rb`: diffusion denoiser model.
- `stable_diffusion/vae.rb`: latent autoencoder.
- `stable_diffusion/sampler.rb`: Euler and Euler-ancestral samplers.
- `stable_diffusion/model_io.rb`: model loading builders.
- `stable_diffusion/__init__.rb`: `StableDiffusion` and `StableDiffusionXL` pipelines.
- `txt2image.rb`: prompt-to-image CLI (writes PPM output).
- `image2image.rb`: image-to-image CLI (reads/writes PPM).
- `test.rb`: synthetic pipeline tests.

## Run

Text-to-image:

```bash
ruby stable_diffusion/txt2image.rb --model sdxl --n-images 2 --steps 2 "A red cube on a table"
```

Image-to-image:

```bash
ruby stable_diffusion/image2image.rb input.ppm "A lit fireplace"
```

## Test

```bash
ruby stable_diffusion/test.rb
```
