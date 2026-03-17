"""Dataset loading and partitioning utilities for federated training."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_window_dataset(data_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the existing window dataset split without changing preprocessing."""
    dataset = {}
    for split in ("train", "val", "test"):
        payload = np.load(f"{data_dir}/{split}.npz")
        dataset[split] = (payload["X"], payload["y"])
    return dataset


def _rng(seed: int) -> np.random.Generator:
    """Create a reproducible NumPy random generator."""
    return np.random.default_rng(seed)


def _build_partition_summary(indices_by_client: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, dict[str, object]]:
    """Create per-client sample and label statistics."""
    summary: dict[str, dict[str, object]] = {}
    for client_id, indices in indices_by_client.items():
        label_counts = Counter(labels[indices].tolist())
        summary[client_id] = {
            "indices": indices.astype(np.int64),
            "sample_count": int(len(indices)),
            "label_distribution": {str(label): int(count) for label, count in sorted(label_counts.items())},
        }
    return summary


def _even_split(indices: np.ndarray, num_clients: int) -> list[np.ndarray]:
    """Split shuffled indices as evenly as possible."""
    return [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(indices, num_clients)]


def _iid_partition(labels: np.ndarray, num_clients: int, seed: int) -> dict[str, np.ndarray]:
    """Create IID partitions by random shuffling."""
    generator = _rng(seed)
    indices = np.arange(len(labels), dtype=np.int64)
    generator.shuffle(indices)
    chunks = _even_split(indices, num_clients)
    return {f"sat_{idx}": chunk for idx, chunk in enumerate(chunks)}


def _dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> dict[str, np.ndarray]:
    """Create a label-skewed Dirichlet partition."""
    generator = _rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    classes = np.unique(labels)

    for label in classes:
        label_indices = np.where(labels == label)[0]
        generator.shuffle(label_indices)
        proportions = generator.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(label_indices)).astype(int)
        chunks = np.split(label_indices, split_points)
        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    for client_id in range(num_clients):
        if not client_indices[client_id]:
            donor = max(range(num_clients), key=lambda idx: len(client_indices[idx]))
            client_indices[client_id].append(client_indices[donor].pop())

    return {
        f"sat_{client_id}": np.asarray(sorted(client_indices[client_id]), dtype=np.int64)
        for client_id in range(num_clients)
    }


def _quantity_skew_partition(labels: np.ndarray, num_clients: int, seed: int) -> dict[str, np.ndarray]:
    """Create quantity-skew partitions with uneven sample counts."""
    generator = _rng(seed)
    indices = np.arange(len(labels), dtype=np.int64)
    generator.shuffle(indices)
    weights = generator.dirichlet(np.full(num_clients, 0.5))
    split_points = (np.cumsum(weights)[:-1] * len(indices)).astype(int)
    chunks = np.split(indices, split_points)
    return {f"sat_{idx}": np.asarray(sorted(chunk), dtype=np.int64) for idx, chunk in enumerate(chunks)}


def _hybrid_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> dict[str, np.ndarray]:
    """Create a hybrid quantity plus label skew partition."""
    generator = _rng(seed)
    quantity_weights = generator.dirichlet(np.full(num_clients, 0.5))
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    classes = np.unique(labels)

    for label in classes:
        label_indices = np.where(labels == label)[0]
        generator.shuffle(label_indices)
        label_weights = generator.dirichlet(np.full(num_clients, alpha))
        mixed_weights = quantity_weights * label_weights
        mixed_weights = mixed_weights / mixed_weights.sum()
        split_points = (np.cumsum(mixed_weights)[:-1] * len(label_indices)).astype(int)
        chunks = np.split(label_indices, split_points)
        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    return {
        f"sat_{client_id}": np.asarray(sorted(client_indices[client_id]), dtype=np.int64)
        for client_id in range(num_clients)
    }


def partition_train_dataset_for_satellites(
    train_npz_path: str,
    num_clients: int,
    mode: str = "dirichlet",
    alpha: float = 0.3,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    """Partition the existing training set into per-satellite local datasets."""
    payload = np.load(train_npz_path)
    labels = payload["y"]

    if mode == "iid":
        indices_by_client = _iid_partition(labels, num_clients, seed)
    elif mode == "dirichlet":
        indices_by_client = _dirichlet_partition(labels, num_clients, alpha, seed)
    elif mode == "quantity_skew":
        indices_by_client = _quantity_skew_partition(labels, num_clients, seed)
    elif mode == "hybrid":
        indices_by_client = _hybrid_partition(labels, num_clients, alpha, seed)
    else:
        raise ValueError(f"Unsupported partition mode: {mode}")

    return _build_partition_summary(indices_by_client, labels)


def create_client_dataloaders(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    partition_map: dict[str, dict[str, object]],
    batch_size: int = 64,
    shuffle: bool = True,
) -> dict[str, DataLoader]:
    """Create per-client dataloaders from the partitioned train split."""
    dataloaders: dict[str, DataLoader] = {}
    for client_id, info in partition_map.items():
        indices = np.asarray(info["indices"], dtype=np.int64)
        x_local = torch.from_numpy(train_features[indices]).float()
        y_local = torch.from_numpy(train_labels[indices]).long()
        dataset = TensorDataset(x_local, y_local)
        dataloaders[client_id] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )
    return dataloaders


def dump_partition_stats(path: str, partition_map: dict[str, dict[str, object]]) -> None:
    """Persist partition statistics for reproducibility."""
    serializable = {
        client_id: {
            "sample_count": int(info["sample_count"]),
            "label_distribution": info["label_distribution"],
        }
        for client_id, info in partition_map.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)
