# Speech Commands KWT (MLX Ruby + DSL)

Ruby port of `mlx-examples/speechcommands`.

## Files

- `kwt.rb`: Keyword Transformer model (`KWT`, `kwt1`, `kwt2`, `kwt3`).
- `dataset.rb`: dataset utility with synthetic MFSC-like splits and optional NPZ loading.
- `main.rb`: training/evaluation loop.
- `test.rb`: local forward/training/smoke tests.

## Data Modes

- `--synthetic` (default): generates MFSC-like random data for quick local testing.
- `--real`: loads preprocessed arrays from `--data-file`.
  - expected keys: `train_audio`, `train_label`, `validation_audio`, `validation_label`, `test_audio`, `test_label`.

## Run

Fast synthetic smoke run:

```bash
ruby speechcommands/main.rb \
  --cpu \
  --synthetic \
  --epochs 1 \
  --batch-size 32 \
  --num-classes 12 \
  --train-samples 256 \
  --val-samples 64 \
  --test-samples 64
```

Run with preprocessed real arrays:

```bash
ruby speechcommands/main.rb \
  --real \
  --data-file speechcommands/data/speechcommands_mfsc.npz \
  --arch kwt1 \
  --epochs 100 \
  --batch-size 256
```

## DSL Notes

- `main.rb` now uses DSL trainer flows for monitoring, patience, and best-checkpoint handling.
- Useful flags for DSL monitoring behavior:
  - `--patience N`
  - `--min-delta N`
  - `--best-ckpt PATH`

## Test

```bash
ruby speechcommands/test.rb
```
