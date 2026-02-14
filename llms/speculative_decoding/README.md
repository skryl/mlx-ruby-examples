# Speculative Decoding (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/llms/speculative_decoding` using T5 models.

## Files

- `model.rb`: T5 encoder/decoder model implementation.
- `decoder.rb`: regular and speculative decoding logic.
- `convert.rb`: converts Hugging Face T5 weights to `.npz`.
- `main.rb`: CLI runner for regular or speculative decoding.
- `test.rb`: local tests for forward, cache parity, and cache truncation.

## Setup

Install Python dependencies used by conversion, config loading, and tokenizer bridges:

```bash
pip install transformers numpy
```

## Convert Models

Convert main model:

```bash
ruby llms/speculative_decoding/convert.rb --model t5-small --dtype float32
```

Convert draft model:

```bash
ruby llms/speculative_decoding/convert.rb --model t5-base --dtype float32
```

This creates files like `t5-small.npz` and `t5-base.npz` in the current directory.

## Run

Speculative decode:

```bash
ruby llms/speculative_decoding/main.rb \
  --model-name t5-base \
  --draft-model-name t5-small \
  --num-draft 5 \
  --max-tokens 100 \
  --delta 0.1
```

Regular decode:

```bash
ruby llms/speculative_decoding/main.rb \
  --model-name t5-base \
  --draft-model-name t5-small \
  --regular-decode
```

## Test

```bash
ruby llms/speculative_decoding/test.rb
```

