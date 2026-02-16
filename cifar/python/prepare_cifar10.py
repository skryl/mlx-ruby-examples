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

def load_split(train):
    try:
        from mlx.data.datasets import load_cifar10

        ds = load_cifar10(root=str(root), train=train)
        stream = ds.to_stream()
        images = []
        labels = []
        for row in stream:
            images.append(row["image"])
            labels.append(row["label"])
        return np.stack(images), np.asarray(labels), "mlx-data"
    except ModuleNotFoundError:
        pass

    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(root), train=train, download=True)
    return np.asarray(dataset.data), np.asarray(dataset.targets), "torchvision"


def dump(train, out_file):
    images, labels, backend = load_split(train)
    np.savez(out_file, images=images, labels=labels)
    return backend


train_backend = dump(True, str(train_out))
test_backend = dump(False, str(test_out))
print(json.dumps({"status": "ok", "backend": {"train": train_backend, "test": test_backend}}))
