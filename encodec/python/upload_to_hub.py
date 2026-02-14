#!/usr/bin/env python3
"""Upload a converted EnCodec checkpoint folder to the Hugging Face Hub."""

import os
import sys
from textwrap import dedent

from huggingface_hub import HfApi, ModelCard, logging


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: upload_to_hub.py <folder> <repo_id> <source_repo>", file=sys.stderr)
        return 2

    folder, repo_id, source_repo = sys.argv[1:4]

    content = dedent(
        f"""
        ---
        language: en
        license: other
        library: mlx
        tags:
          - mlx
        ---

        Converted EnCodec checkpoint for MLX Ruby.

        Source model: https://huggingface.co/{source_repo}
        """
    ).strip() + "\n"

    os.makedirs(folder, exist_ok=True)
    card = ModelCard(content)
    card.save(os.path.join(folder, "README.md"))

    logging.set_verbosity_info()
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
