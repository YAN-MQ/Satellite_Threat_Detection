"""Configuration objects and formal presets for OrbitShield_FL training."""

from __future__ import annotations

from dataclasses import dataclass, field


FULL_METHOD_PRESET: dict[str, float | int] = {
    "beta": 0.1,
    "beta_floor": 0.05,
    "lambda_s": 0.1,
    "rho": 0.5,
    "mu": 0.8,
    "global_momentum": 0.1,
    "warmup_rounds": 2,
}


METHOD_PRESETS: dict[str, dict[str, float | int]] = {
    "single": {
        "beta": 0.0,
        "beta_floor": 0.0,
        "lambda_s": 0.0,
        "rho": 0.0,
        "mu": 0.8,
        "global_momentum": 0.0,
        "warmup_rounds": 0,
    },
    "fedavg": {
        "beta": 0.0,
        "beta_floor": 0.0,
        "lambda_s": 0.0,
        "rho": 0.0,
        "mu": 0.8,
        "global_momentum": 0.0,
        "warmup_rounds": 0,
    },
    "intra_only": {
        "beta": 0.0,
        "beta_floor": 0.0,
        "lambda_s": 0.1,
        "rho": 0.0,
        "mu": 0.8,
        "global_momentum": 0.0,
        "warmup_rounds": 0,
    },
    "intra_gossip": {
        "beta": 0.1,
        "beta_floor": 0.05,
        "lambda_s": 0.1,
        "rho": 0.5,
        "mu": 0.8,
        "global_momentum": 0.0,
        "warmup_rounds": 0,
    },
    "full": FULL_METHOD_PRESET.copy(),
}


@dataclass
class FederatedConfig:
    """Runtime configuration for the federated training simulator."""

    dataset: str = "cicids17"
    data_dir: str = "../dataset_cicids17"
    output_dir: str = "experiments/OrbitShield_FL/cicids17"
    topology_backend: str = "heuristic"
    ns3_trace_dir: str | None = None
    method: str = "full"
    num_clients: int = 12
    num_planes: int = 3
    rounds: int = 20
    local_epochs: int = 1
    max_local_batches: int | None = None
    batch_size: int = 512
    input_dim: int = 18
    num_classes: int = 3
    hidden_dim: int = 64
    bidirectional: bool = False
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-2
    init_checkpoint: str | None = "checkpoints_gru/cicids17_gru_best.pt"
    max_samples: int | None = None
    eval_subset_size: int | None = None
    test_subset_size: int | None = None
    eval_every: int = 1
    test_every: int = 0
    partition_mode: str = "dirichlet"
    dirichlet_alpha: float = 0.3
    beta: float = float(FULL_METHOD_PRESET["beta"])
    beta_floor: float = float(FULL_METHOD_PRESET["beta_floor"])
    lambda_s: float = float(FULL_METHOD_PRESET["lambda_s"])
    rho: float = float(FULL_METHOD_PRESET["rho"])
    mu: float = float(FULL_METHOD_PRESET["mu"])
    global_momentum: float = float(FULL_METHOD_PRESET["global_momentum"])
    warmup_rounds: int = int(FULL_METHOD_PRESET["warmup_rounds"])
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
        if self.topology_backend not in {"heuristic", "ns3"}:
            raise ValueError(f"Unsupported topology backend: {self.topology_backend}")
        if self.topology_backend == "ns3" and not self.ns3_trace_dir:
            raise ValueError("ns3_trace_dir must be provided when topology_backend='ns3'")
