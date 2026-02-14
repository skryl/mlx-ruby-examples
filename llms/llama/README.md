# Llama (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/llms/llama`.

## Files

- `llama.rb`: Llama model + generation CLI.
- `convert.rb`: converts torch checkpoints to MLX `.npz`.
- `test.rb`: local model behavior tests.

## Setup

Install Python dependencies used by conversion and tokenization bridges:

```bash
pip install mlx torch transformers sentencepiece numpy
```

## Convert Weights

```bash
ruby llms/llama/convert.rb \
  --torch-path /path/to/llama-or-tinyllama \
  --mlx-path llms/llama/mlx_model \
  --model-name llama
```

Optional quantization:

```bash
ruby llms/llama/convert.rb \
  --torch-path /path/to/llama-or-tinyllama \
  --mlx-path llms/llama/mlx_model_q4 \
  --model-name llama \
  --quantize \
  --q-group-size 64 \
  --q-bits 4
```

Use `--model-name tiny_llama` for TinyLlama-style checkpoints.

## Run

```bash
ruby llms/llama/llama.rb \
  --model-path llms/llama/mlx_model \
  --prompt "In the beginning the Universe was created." \
  --max-tokens 100 \
  --temp 0.0
```

## Test

```bash
ruby llms/llama/test.rb
```
