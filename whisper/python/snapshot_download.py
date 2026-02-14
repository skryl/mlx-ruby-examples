#!/usr/bin/env python3
"""Download whisper model snapshot and print local path."""

import sys

from huggingface_hub import snapshot_download


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: snapshot_download.py <repo_id>", file=sys.stderr)
        return 2

    repo_id = sys.argv[1]
    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.npz",
            "pytorch_model.bin",
            "*.tiktoken",
        ],
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
