#!/usr/bin/env python3
"""Extract a torch checkpoint into an NPZ file for MLX Ruby."""

import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_torch_weights.py <pytorch_model.bin> <out.npz>", file=sys.stderr)
        return 2

    src = Path(sys.argv[1])
    out = Path(sys.argv[2])

    if not src.exists():
        print(f"missing input: {src}", file=sys.stderr)
        return 1

    state = torch.load(src, map_location="cpu", weights_only=True)
    arrays = {k: v.detach().cpu().numpy() for k, v in state.items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
