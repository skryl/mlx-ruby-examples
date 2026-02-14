import json
import os
import tarfile
from pathlib import Path
from urllib import request

import numpy as np
import scipy.sparse as sparse


def download_cora(root_dir: Path):
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    cora_dir = root_dir / "cora"
    if cora_dir.exists():
        return

    root_dir.mkdir(parents=True, exist_ok=True)
    archive_path = root_dir / "cora.tgz"
    request.urlretrieve(url, archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root_dir)
    archive_path.unlink(missing_ok=True)


def enumerate_labels(labels):
    label_map = {v: e for e, v in enumerate(set(labels))}
    return np.array([label_map[label] for label in labels], dtype=np.int32)


def normalize_adjacency(adj):
    adj = adj + sparse.eye(adj.shape[0])

    node_degrees = np.array(adj.sum(1))
    node_degrees = np.power(node_degrees, -0.5).flatten()
    node_degrees[np.isinf(node_degrees)] = 0.0
    node_degrees[np.isnan(node_degrees)] = 0.0
    degree_matrix = sparse.diags(node_degrees, dtype=np.float32)

    return degree_matrix @ adj @ degree_matrix


def load_and_prepare(nodes_path: Path, edges_path: Path):
    raw_nodes_data = np.genfromtxt(str(nodes_path), dtype="str")
    raw_node_ids = raw_nodes_data[:, 0].astype("int32")
    raw_node_labels = raw_nodes_data[:, -1]
    labels_enumerated = enumerate_labels(raw_node_labels)
    node_features = sparse.csr_matrix(raw_nodes_data[:, 1:-1], dtype="float32")

    ids_ordered = {raw_id: order for order, raw_id in enumerate(raw_node_ids)}
    raw_edges_data = np.genfromtxt(str(edges_path), dtype="int32")
    edges_ordered = np.array(
        list(map(ids_ordered.get, raw_edges_data.flatten())), dtype="int32"
    ).reshape(raw_edges_data.shape)

    adj = sparse.coo_matrix(
        (np.ones(edges_ordered.shape[0]), (edges_ordered[:, 0], edges_ordered[:, 1])),
        shape=(labels_enumerated.shape[0], labels_enumerated.shape[0]),
        dtype=np.float32,
    )

    adj = adj + adj.T.multiply(adj.T > adj)
    adj = normalize_adjacency(adj)

    features = node_features.toarray().astype(np.float32)
    labels = labels_enumerated.astype(np.int32)
    adjacency = adj.toarray().astype(np.float32)
    return features, labels, adjacency


def main():
    out_file = Path(sys.argv[1]).expanduser()
    data_root = Path(sys.argv[2]).expanduser()

    if out_file.exists():
        print(json.dumps({"status": "exists", "path": str(out_file)}))
        return

    download_cora(data_root)
    nodes_path = data_root / "cora" / "cora.content"
    edges_path = data_root / "cora" / "cora.cites"
    features, labels, adjacency = load_and_prepare(nodes_path, edges_path)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_file,
        features=features,
        labels=labels,
        adjacency=adjacency,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "path": str(out_file),
                "features_shape": list(features.shape),
                "labels_shape": list(labels.shape),
                "adj_shape": list(adjacency.shape),
            }
        )
    )


if __name__ == "__main__":
    import sys

    main()
