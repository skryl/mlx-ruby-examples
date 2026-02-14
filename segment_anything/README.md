# Segment Anything (MLX Ruby + DSL)

Ruby port of `mlx-examples/segment_anything` with a synthetic-first test path.

## Files

- `common.rb`: shared MLP and 2D layer norm blocks.
- `image_encoder.rb`: ViT-style image encoder.
- `prompt_encoder.rb`: point/box/mask prompt encoding.
- `transformer.rb`: two-way transformer decoder blocks.
- `mask_decoder.rb`: mask tokens, upscaling, and IoU head.
- `sam.rb`: SAM model assembly, preprocess/postprocess, and model loaders.
- `predictor.rb`: predictor API for repeated prompt inference on one image.
- `automatic_mask_generator.rb`: automatic mask generation over point grids.
- `utils/transforms.rb`: resize and coordinate transform utilities.
- `utils/amg.rb`: mask/box/RLE helper utilities.
- `main.rb`: CLI to run mask generation on synthetic or NPZ input image.
- `convert.rb`: Ruby wrapper for HF->MLX conversion bridge.
- `python/convert.py`: conversion bridge script.
- `test.rb`: synthetic model/predictor/generator tests.

## Setup

Install Python dependencies for conversion bridges:

```bash
pip install -r segment_anything/requirements.txt
```

## Run

Synthetic mask generation smoke run:

```bash
ruby segment_anything/main.rb --output segment_anything/output
```

Run on NPZ input (expects key `image` with shape HxWx3):

```bash
ruby segment_anything/main.rb --input-npz path/to/image.npz --output segment_anything/output
```

Convert HF SAM weights:

```bash
ruby segment_anything/convert.rb --hf-path facebook/sam-vit-base --mlx-path segment_anything/sam-vit-base
```

## Test

```bash
ruby segment_anything/test.rb
```
