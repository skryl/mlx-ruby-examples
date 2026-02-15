# Graph Convolutional Network (MLX Ruby + DSL)

Ruby port of `mlx-examples/gcn`.

## Files

- `gcn.rb`: `GCNLayer` and `GCN` model definitions.
- `datasets.rb`: synthetic graph generation plus real Cora loading bridge.
- `main.rb`: training script with early stopping.
- `test.rb`: local model/data/training-step smoke tests.
- `python/prepare_cora.py`: prepares normalized Cora arrays into `.npz` for Ruby loading.

## Setup

Install Python dependencies used only for real Cora preprocessing:

```bash
pip install -r gcn/requirements.txt
```

## Run

Fast synthetic run (default mode):

```bash
ruby gcn/main.rb --cpu --synthetic --epochs 20
```

Train on real Cora:

```bash
ruby gcn/main.rb --real --data-root gcn/data --epochs 100
```

The first real run prepares `gcn/data/cora_processed.npz`.

## DSL Notes

- Training/validation orchestration in `main.rb` is DSL-first (`MLX::DSL::Trainer` + split plans/dataflow profiles).
- Best-checkpoint behavior is managed by DSL artifact policy (no manual best-metric loop).
- Optional artifact/resume flags:
  - `--run-bundle PATH`
  - `--resume-from SOURCE`
  - `--best-ckpt PATH`

## Test

```bash
ruby gcn/test.rb
```
