"""Dataset profile helpers for switching between CICIDS17 and STI datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProfile:
    """Resolved dataset-specific training defaults."""

    name: str
    data_dir: str
    input_dim: int
    num_classes: int
    seq_len: int
    output_checkpoint: str
    init_checkpoint: str | None
    class_names: list[str]


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "cicids17": DatasetProfile(
        name="cicids17",
        data_dir="../dataset_cicids17",
        input_dim=18,
        num_classes=3,
        seq_len=10,
        output_checkpoint="checkpoints_gru/cicids17_gru_best.pt",
        init_checkpoint="checkpoints_gru/cicids17_gru_best.pt",
        class_names=["Benign", "DDoS", "PortScan"],
    ),
    "sti": DatasetProfile(
        name="sti",
        data_dir="../dataset_sti",
        input_dim=20,
        num_classes=8,
        seq_len=1,
        output_checkpoint="checkpoints_gru/sti_gru_best.pt",
        init_checkpoint=None,
        class_names=[
            "Benign",
            "Signal Disruption",
            "UDP flood",
            "Jamming",
            "Bruteforce",
            "Infiltration",
            "DoS",
            "DDoS",
        ],
    ),
}


def get_dataset_profile(dataset: str) -> DatasetProfile:
    """Return the predefined training profile for a dataset name."""
    if dataset not in DATASET_PROFILES:
        raise ValueError(f"Unsupported dataset profile: {dataset}")
    return DATASET_PROFILES[dataset]
