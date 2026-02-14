# T5 (MLX Ruby + DSL)

Ruby port of `mlx-examples/t5`.

## Files

- `model.rb`: T5 model, tokenizer bridge, config bridge, and generation helpers.
- `main.rb`: inference CLI (encode-only or generation).
- `convert.rb`: converts Hugging Face T5 checkpoint to `.npz` MLX-Ruby weights.
- `hf_t5.rb`: utility wrapper for Hugging Face reference behavior.
- `test.rb`: local model/cache/generation tests.
- `python/*.py`: dedicated Python bridge scripts for config/tokenizer/HF conversion and utility parity.

## Setup

Install Python dependencies for bridges and conversion:

```bash
pip install -r t5/requirements.txt
```

## Convert Weights

```bash
ruby t5/convert.rb --model t5-small --dtype float32
```

This writes `t5-small.npz` by default.

## Run

Generate text:

```bash
ruby t5/main.rb --model t5-small --prompt "translate English to German: A tasty apple"
```

Encoder-only mode:

```bash
ruby t5/main.rb --model t5-small --encode-only --prompt "translate English to German: That is good."
```

Reference Hugging Face utility:

```bash
ruby t5/hf_t5.rb --model t5-small
```

## Test

```bash
ruby t5/test.rb
```
