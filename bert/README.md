# BERT (MLX Ruby + DSL)

This directory contains a Ruby/DSL port of the MLX Python BERT example from
`mlx-examples/bert`.

## Files

- `model.rb`: DSL-based BERT implementation and CLI runner.
- `convert.rb`: converts Hugging Face BERT weights to MLX-compatible `.npz`.
- `hf_bridge.rb`: Ruby utility bridge for config/tokenization/torch reference outputs.
- `test.rb`: parity test against Hugging Face torch outputs.

## Setup

Install Python dependencies used by the Ruby utilities for conversion,
tokenization, and parity tests:

```bash
pip install -r bert/requirements.txt
```

## Convert Weights

```bash
ruby bert/convert.rb \
  --bert-model bert-base-uncased \
  --mlx-model bert/weights/bert-base-uncased.npz
```

This writes:

- `bert/weights/bert-base-uncased.npz`
- `bert/weights/bert-base-uncased.config.json`

## Run (Ruby)

```bash
ruby bert/model.rb \
  --bert-model bert-base-uncased \
  --mlx-model bert/weights/bert-base-uncased.npz \
  --config-path bert/weights/bert-base-uncased.config.json \
  --text "This is an example of BERT working in MLX Ruby."
```

The runner prints sequence and pooled output shapes.

## Parity Test

Default test path is synthetic/offline and does not require Python packages:

```bash
ruby bert/test.rb
```

Run Hugging Face parity integration explicitly:

```bash
ruby bert/test.rb \
  --integration \
  --bert-model bert-base-uncased \
  --mlx-model bert/weights/bert-base-uncased.npz \
  --config-path bert/weights/bert-base-uncased.config.json \
  --text "This is an example of BERT working in MLX Ruby."
```

It compares Ruby model outputs against Hugging Face torch outputs.
