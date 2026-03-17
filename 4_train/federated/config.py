"""Configuration objects for federated constellation training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FederatedConfig:
    """Runtime configuration for the federated training simulator."""

    data_dir: str = "../dataset_window"
    output_dir: str = "experiments_window/federated/OrbitShield_FL"
    method: str = "full"
    num_clients: int = 12
    num_planes: int = 3
    rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 512
    input_dim: int = 18
    num_classes: int = 3
    hidden_dim: int = 64
    bidirectional: bool = False
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-2
    init_checkpoint: str | None = "checkpoints_gru/window_gru_best.pt"
    partition_mode: str = "dirichlet"
    dirichlet_alpha: float = 0.3
    beta: float = 0.1
    beta_floor: float = 0.05
    lambda_s: float = 0.1
    rho: float = 0.5
    mu: float = 0.8
    global_momentum: float = 0.1
    warmup_rounds: int = 2
    r_min: float = 0.1
    seed: int = 42
    device: str | None = None
    intra_plane_success_prob: float = 0.98
    inter_plane_success_prob: float = 0.75
    inter_plane_contact_period: int = 4
    inter_plane_contact_duration: int = 2
    packet_loss_prob: float = 0.05
    link_delay_mean: float = 25.0
    stable_score_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    class_names: list[str] = field(default_factory=lambda: ["Benign", "DDoS", "PortScan"])

    @property
    def sats_per_plane(self) -> int:
        """Return the number of satellites per plane."""
        return self.num_clients // self.num_planes

    def validate(self) -> None:
        """Validate basic configuration consistency."""
        if self.num_clients % self.num_planes != 0:
            raise ValueError("num_clients must be divisible by num_planes")
        if self.method not in {"single", "fedavg", "intra_only", "intra_gossip", "full"}:
            raise ValueError(f"Unsupported method: {self.method}")
