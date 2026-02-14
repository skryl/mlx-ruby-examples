# Mixtral (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/llms/mixtral`.

## Files

- `mixtral.rb`: Mixtral MoE model + generation CLI.
- `convert.rb`: converts Mixtral consolidated torch shards to MLX `.npz` shards.
- `params.json`: default architecture params used by converter.
- `test.rb`: local model behavior tests.

## Setup

Install Python dependencies used by conversion and tokenization bridges:

```bash
pip install mlx torch numpy sentencepiece
```

## Get Model

Example from Hugging Face:

```bash
export MIXTRAL_MODEL=Mixtral-8x7B-v0.1
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mistralai/${MIXTRAL_MODEL}
cd ${MIXTRAL_MODEL}
git lfs pull --include "consolidated.*.pt"
git lfs pull --include "tokenizer.model"
```

## Convert Weights

```bash
ruby llms/mixtral/convert.rb \
  --torch-path /path/to/${MIXTRAL_MODEL} \
  --mlx-path llms/mixtral/mlx_model
```

Optional quantization:

```bash
ruby llms/mixtral/convert.rb \
  --torch-path /path/to/${MIXTRAL_MODEL} \
  --mlx-path llms/mixtral/mlx_model_q4 \
  --quantize \
  --q-group-size 64 \
  --q-bits 4
```

## Run

```bash
ruby llms/mixtral/mixtral.rb \
  --model-path llms/mixtral/mlx_model \
  --prompt "[INST] Write a quicksort in Ruby [/INST]" \
  --max-tokens 100 \
  --temp 0.0
```

## Test

```bash
ruby llms/mixtral/test.rb
```

