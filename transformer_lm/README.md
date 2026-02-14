# Transformer LM (MLX Ruby + DSL)

Ruby port of `mlx-examples/transformer_lm`.

## Files

- `datasets.rb`: dataset loaders for `ptb`, `wikitext2`, `wikitext103`, and `enwik8`.
- `main.rb`: decoder-only Transformer LM training script.
- `test.rb`: local tests for dataset utilities, sampling, forward, loss, and training step.

## Run

Fast synthetic smoke run:

```bash
ruby transformer_lm/main.rb \
  --synthetic \
  --num-iters 20 \
  --steps-per-report 5 \
  --steps-per-eval 10 \
  --batch-size 4 \
  --context-size 64 \
  --num-blocks 2 \
  --dim 128 \
  --num-heads 4
```

Train on PTB:

```bash
ruby transformer_lm/main.rb \
  --dataset ptb \
  --batch-size 2 \
  --context-size 1024 \
  --num-blocks 12 \
  --dim 1024 \
  --num-heads 16
```

By default, this runs on CPU. Add `--gpu` to run on the Metal backend.

## Test

```bash
ruby transformer_lm/test.rb
```
