# CLIP (MLX Ruby + DSL)

Ruby port of `mlx-examples/clip`.

## Files

- `model.rb`: lightweight CLIP-style dual-encoder model.
- `tokenizer.rb`: simple byte tokenizer.
- `image_processor.rb`: resize + normalize image preprocessing.
- `clip.rb`: text/image similarity inference script.
- `linear_probe.rb`: synthetic linear-probe training script.
- `convert.rb`: wrapper around the original CLIP HF-to-MLX conversion utility.
- `hf_preproc.rb`: wrapper around original Hugging Face preprocessing parity script.
- `test.rb`: local tokenizer/model/training/smoke tests.

## Setup

Install optional Python dependencies used by conversion/HF wrappers:

```bash
pip install -r clip/requirements.txt
```

## Run

Quick synthetic CLIP run:

```bash
ruby clip/clip.rb --text "a photo of a cat" --text "a photo of a dog" --image-size 32 --patch-size 8
```

Linear probe smoke run:

```bash
ruby clip/linear_probe.rb --samples 128 --classes 5 --image-size 32 --epochs 3
```

Convert HF CLIP weights (wrapper to upstream converter):

```bash
ruby clip/convert.rb --hf-repo openai/clip-vit-base-patch32 --mlx-path clip/mlx_model
```

## Test

```bash
ruby clip/test.rb
```
