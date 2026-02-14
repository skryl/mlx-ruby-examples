# LLaVA (MLX Ruby + DSL)

Ruby port of `mlx-examples/llava`.

## Files

- `language.rb`: LLaMA-style language backbone.
- `vision.rb`: CLIP-vision tower and embeddings.
- `llava.rb`: multimodal projector + LLaVA model wiring and weight loading.
- `generate.rb`: image+prompt generation CLI.
- `test.rb`: synthetic local tests for merge/cache/forward behavior.
- `python/processor_bridge.py`: Hugging Face `AutoProcessor` bridge for preprocessing/decoding.
- `python/snapshot_download.py`: model snapshot download bridge.

## Setup

Install Python dependencies used by the bridges:

```bash
pip install -r llava/requirements.txt
```

## Run

Generate from an image and prompt:

```bash
ruby llava/generate.rb \
  --model llava-hf/llava-1.5-7b-hf \
  --image "http://images.cocodataset.org/val2017/000000039769.jpg" \
  --prompt "USER: <image>\\nWhat are these?\\nASSISTANT:" \
  --max-tokens 128 \
  --temp 0.0
```

Note: full LLaVA checkpoints are large; this command may download model files on first run.

## Test

```bash
ruby llava/test.rb
```
