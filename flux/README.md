# FLUX (MLX Ruby + DSL)

Ruby port of `mlx-examples/flux` with full file coverage and synthetic-friendly defaults.

## Files

- `flux/flux.rb`: `FluxPipeline` generation/training API.
- `flux/model.rb`: core transformer model (`Flux`, `FluxParams`).
- `flux/layers.rb`: attention/modulation blocks and embeddings.
- `flux/autoencoder.rb`: latent autoencoder.
- `flux/clip.rb`: CLIP text encoder.
- `flux/t5.rb`: T5 text encoder.
- `flux/tokenizers.rb`: CLIP/T5 tokenizers.
- `flux/sampler.rb`: FLUX sampler.
- `flux/lora.rb`: LoRA linear module.
- `flux/datasets.rb`: local/HF dataset loaders.
- `flux/trainer.rb`: dataset encoding and training iterator.
- `flux/utils.rb`: model specs/loaders/config utilities.
- `txt2image.rb`: prompt-to-image CLI.
- `generate_interactive.rb`: interactive generation CLI.
- `dreambooth.rb`: LoRA finetuning script.
- `test.rb`: end-to-end synthetic tests.
- `python/hf_download.py`: HF download bridge used by `flux/utils.rb`.

## Run

Text-to-image:

```bash
ruby flux/txt2image.rb --model schnell --n-images 2 --image-size 256x256 "A photo of an astronaut riding a horse on Mars"
```

Interactive session:

```bash
ruby flux/generate_interactive.rb --model schnell --output flux/out.ppm
```

Dreambooth finetuning:

```bash
ruby flux/dreambooth.rb \
  --progress-prompt "A photo of a dog on the beach" \
  --iterations 50 \
  --batch-size 1 \
  path/to/dataset
```

## Test

```bash
ruby flux/test.rb
```
