# MusicGen (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/musicgen`.

## Files

- `musicgen.rb`: MusicGen model, cache, transformer blocks, sampling helpers, and pretrained-loading utilities.
- `generate.rb`: text-to-audio generation CLI.
- `utils.rb`: audio saving utility.
- `encodec.rb`: local wrapper for the Ruby EnCodec implementation.
- `t5.rb`: local wrapper for the Ruby T5 implementation.
- `test.rb`: local model/cache/sampling/generation tests.
- `python/snapshot_download.py`: Hugging Face snapshot bridge.
- `python/extract_state_dict.py`: converts MusicGen `state_dict.bin` to `.npz` for Ruby loading.

## Setup

Install optional Python dependencies for bridge scripts:

```bash
pip install -r musicgen/requirements.txt
```

## Generate

```bash
ruby musicgen/generate.rb \
  --model facebook/musicgen-medium \
  --text "happy rock" \
  --max-steps 200 \
  --output-path musicgen/out.wav
```

## Test

```bash
ruby musicgen/test.rb
```
