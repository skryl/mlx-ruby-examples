#!/usr/bin/env python3

import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
path = snapshot_download(
    repo_id=repo_id,
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.npz",
        "tokenizer.model",
        "*.tiktoken",
    ],
)
print(path)
