# CIFAR ResNet (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/cifar`.

## Files

- `resnet.rb`: CIFAR ResNet model definitions (`resnet20` ... `resnet1202`).
- `dataset.rb`: CIFAR-10 batch iterator utility with augmentation, normalization, and optional Python data bridge.
- `main.rb`: training script.
- `test.rb`: local forward/training-step/data-iterator tests.

## Setup

Install optional Python dependencies used by the CIFAR-10 data bridge:

```bash
pip install -r cifar/requirements.txt
```

If these are not installed, you can still run synthetic-data training with `--synthetic`.

## Run

Quick synthetic smoke run:

```bash
ruby cifar/main.rb \
  --synthetic \
  --epochs 1 \
  --batch-size 64 \
  --train-samples 2048 \
  --test-samples 512 \
  --arch resnet20
```

Using real CIFAR-10 with Python bridge/cache:

```bash
ruby cifar/main.rb \
  --data-root cifar/data \
  --epochs 30 \
  --batch-size 256 \
  --arch resnet20
```

The first run prepares `cifar10_train.npz` and `cifar10_test.npz` in `--data-root`.

To run on CPU:

```bash
ruby cifar/main.rb --cpu --synthetic --epochs 1
```

## Test

```bash
ruby cifar/test.rb
```
