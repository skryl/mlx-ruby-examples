# CVAE on MNIST (MLX Ruby + DSL)

Ruby port of `mlx-examples/cvae`.

## Files

- `dataset.rb`: MNIST/Fashion-MNIST iterators with on-the-fly resize to `HxW`.
- `vae.rb`: convolutional VAE model (`Encoder`, `Decoder`, `CVAE`).
- `main.rb`: training script with reconstruction/sample export.
- `test.rb`: local model/data/training-step tests.

## Setup

Install Python dependency used by the MNIST dataset bridge:

```bash
pip install -r cvae/requirements.txt
```

## Run

Fast synthetic smoke run:

```bash
ruby cvae/main.rb \
  --cpu \
  --synthetic \
  --epochs 1 \
  --batch-size 32 \
  --train-size 512 \
  --test-size 128 \
  --save-dir cvae/models_smoke
```

Train with real MNIST:

```bash
ruby cvae/main.rb \
  --dataset mnist \
  --data-root cvae/data \
  --epochs 50 \
  --save-dir cvae/models
```

Train with Fashion-MNIST:

```bash
ruby cvae/main.rb \
  --dataset fashion_mnist \
  --data-root cvae/data \
  --epochs 50 \
  --save-dir cvae/models_fashion
```

Saved reconstructions/samples are written as `.pgm` images by default. Use `--no-save-images` to disable export.

## DSL Notes

- Training in `main.rb` is orchestrated with `MLX::DSL::Trainer` hooks.
- Checkpoint lifecycle is managed with DSL artifact policy under `cvae/models/checkpoints/`.

## Test

```bash
ruby cvae/test.rb
```
