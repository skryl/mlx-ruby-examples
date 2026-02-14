# LoRA / QLoRA Finetuning (MLX Ruby + DSL)

Ruby port of `mlx-examples/lora`.

## Files

- `models.rb`: LLaMA-style causal LM modules plus `LoRALinear`.
- `utils.rb`: model/tokenizer loading helpers, model saving, and generation.
- `lora.rb`: finetuning / evaluation / generation CLI.
- `convert.rb`: converts base LLaMA-style checkpoints for LoRA workflows.
- `fuse.rb`: fuses trained LoRA adapters into a standalone model.
- `data/wikisql.rb`: WikiSQL preprocessing utility.
- `test.rb`: local synthetic training/fusion tests.
- `python/tokenizer_bridge.py`: dedicated tokenizer bridge for Hugging Face tokenizers.

## Setup

Install optional Python dependency used by tokenizer bridging:

```bash
pip install -r lora/requirements.txt
```

## Train (Synthetic Base Model)

```bash
ruby lora/lora.rb \
  --synthetic-model \
  --train \
  --data mlx-examples/lora/data \
  --iters 100 \
  --batch-size 4 \
  --lora-layers 1 \
  --adapter-file lora/adapters.npz
```

## Generate

```bash
ruby lora/lora.rb \
  --synthetic-model \
  --adapter-file lora/adapters.npz \
  --prompt "select the average price by category"
```

## Fuse

```bash
ruby lora/fuse.rb \
  --synthetic-model \
  --adapter-file lora/adapters.npz \
  --save-path lora/fused_model
```

## Test

```bash
ruby lora/test.rb
```
