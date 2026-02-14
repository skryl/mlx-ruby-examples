import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]).expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)
train_out = root / "cifar10_train.npz"
test_out = root / "cifar10_test.npz"

if train_out.exists() and test_out.exists():
    print(json.dumps({"status": "exists"}))
    raise SystemExit(0)

from mlx.data.datasets import load_cifar10


def dump(train, out_file):
    ds = load_cifar10(root=str(root), train=train)
    stream = ds.to_stream()
    images = []
    labels = []
    for row in stream:
        images.append(row["image"])
        labels.append(row["label"])
    np.savez(out_file, images=np.stack(images), labels=np.asarray(labels))


dump(True, str(train_out))
dump(False, str(test_out))
print(json.dumps({"status": "ok"}))
