# Normalizing Flow (MLX Ruby + DSL)

Ruby port of `mlx-examples/normalizing_flow` (RealNVP).

## Files

- `distributions.rb`: Normal distribution utilities.
- `bijectors.rb`: affine and masked coupling bijectors.
- `flows.rb`: MLP conditioner and RealNVP model.
- `main.rb`: training script on a generated two-moons dataset.
- `test.rb`: local shape/invertibility/training/smoke tests.

## Run

Quick CPU smoke run:

```bash
ruby normalizing_flow/main.rb \
  --cpu \
  --n-steps 200 \
  --n-batch 64 \
  --n-transforms 6 \
  --output normalizing_flow/samples.npz
```

This writes sampled arrays to `normalizing_flow/samples.npz` with keys:

- `transform_0` ... `transform_N` for intermediate transformed samples.
- `original` for the training two-moons data.

## DSL Notes

- `main.rb` uses DSL trainer + data pipelines (split plans/dataflow profiles) while preserving the original RealNVP objective.
- `flows.rb` model declarations now use `MLX::DSL::Model` macros for conditioner construction ergonomics.

## Test

```bash
ruby normalizing_flow/test.rb
```
