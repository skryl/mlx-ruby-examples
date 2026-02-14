# Mistral (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/llms/mistral`.

## Files

- `mistral.rb`: Mistral model + generation CLI.
- `convert.rb`: converts Mistral torch weights to MLX `.npz`.
- `test.rb`: local tests plus optional integration token test.

## Setup

Install Python dependencies used by conversion and tokenization bridges:

```bash
pip install mlx torch numpy sentencepiece
```

## Get Model

Example for `mistral-7B-v0.1`:

```bash
curl -O https://models.mistralcdn.com/mistral-7b-v0-1/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```

## Convert Weights

```bash
ruby llms/mistral/convert.rb \
  --torch-path mistral-7B-v0.1 \
  --mlx-path llms/mistral/mlx_model
```

Optional quantization:

```bash
ruby llms/mistral/convert.rb \
  --torch-path mistral-7B-v0.1 \
  --mlx-path llms/mistral/mlx_model_q4 \
  --quantize \
  --q-group-size 64 \
  --q-bits 4
```

## Run

```bash
ruby llms/mistral/mistral.rb \
  --model-path llms/mistral/mlx_model \
  --prompt "In the beginning the Universe was created." \
  --max-tokens 100 \
  --temp 0.0
```

## Test

Fast local tests:

```bash
ruby llms/mistral/test.rb
```

Optional checkpoint integration test (expects converted weights):

```bash
ruby llms/mistral/test.rb \
  --integration-model-path llms/mistral/mlx_model
```

