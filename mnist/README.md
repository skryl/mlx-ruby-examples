# MNIST MLP (MLX Ruby + DSL)

Ruby port of `mlx-examples/mnist` with a Python bridge for dataset preparation.

## Files

- `mnist.rb`: MNIST/Fashion-MNIST dataset loader and synthetic fallback.
- `main.rb`: MLP training script.
- `test.rb`: local forward/training/data tests.
- `python/prepare_dataset.py`: Python bridge script that downloads/parses IDX files into `.npz`.

## Setup

Install Python dependency for the dataset bridge:

```bash
pip install -r mnist/requirements.txt
```

## Run

Train on synthetic data (fast smoke run):

```bash
ruby mnist/main.rb --synthetic --epochs 1 --batch-size 128 --train-size 4096 --test-size 1024
```

Train on MNIST (downloads and caches into `--data-root`):

```bash
ruby mnist/main.rb --dataset mnist --data-root mnist/data --epochs 10
```

Train on Fashion-MNIST:

```bash
ruby mnist/main.rb --dataset fashion_mnist --data-root mnist/data --epochs 10
```

Run on GPU:

```bash
ruby mnist/main.rb --gpu --synthetic --epochs 1
```

## Test

```bash
ruby mnist/test.rb
```
