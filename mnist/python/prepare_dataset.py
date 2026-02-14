import gzip
import json
import os
import sys
from urllib import request

import numpy as np


def prepare(save_file: str, base_url: str):
    names = [
        ("training_images", "train-images-idx3-ubyte.gz"),
        ("test_images", "t10k-images-idx3-ubyte.gz"),
        ("training_labels", "train-labels-idx1-ubyte.gz"),
        ("test_labels", "t10k-labels-idx1-ubyte.gz"),
    ]

    out_dir = os.path.dirname(save_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    tmp_files = []
    for _, name in names:
        out_file = os.path.join(out_dir, name)
        if not os.path.exists(out_file):
            request.urlretrieve(base_url + name, out_file)
        tmp_files.append(out_file)

    dataset = {}
    for key, name in names[:2]:
        out_file = os.path.join(out_dir, name)
        with gzip.open(out_file, "rb") as f:
            dataset[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

    for key, name in names[2:]:
        out_file = os.path.join(out_dir, name)
        with gzip.open(out_file, "rb") as f:
            dataset[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    dataset["training_images"] = dataset["training_images"].astype(np.float32) / 255.0
    dataset["test_images"] = dataset["test_images"].astype(np.float32) / 255.0
    dataset["training_labels"] = dataset["training_labels"].astype(np.int32)
    dataset["test_labels"] = dataset["test_labels"].astype(np.int32)

    np.savez(
        save_file,
        training_images=dataset["training_images"],
        training_labels=dataset["training_labels"],
        test_images=dataset["test_images"],
        test_labels=dataset["test_labels"],
    )


if __name__ == "__main__":
    save_file = sys.argv[1]
    base_url = sys.argv[2]

    if os.path.exists(save_file):
        print(json.dumps({"status": "exists", "path": save_file}))
        raise SystemExit(0)

    prepare(save_file, base_url)
    print(json.dumps({"status": "ok", "path": save_file}))
