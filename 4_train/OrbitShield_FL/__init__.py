"""OrbitShield_FL federated learning framework for multi-satellite threat prediction."""

from .client import FederatedClient
from .config import FederatedConfig
from .config import FULL_METHOD_PRESET, METHOD_PRESETS
from .partition import (
    create_client_dataloaders,
    load_window_dataset,
    partition_train_dataset_for_satellites,
)
from .serverless_orchestrator import (
    ServerlessOrchestrator,
    run_federated_training,
    train_one_federated_round,
)

__all__ = [
    "FederatedClient",
    "FederatedConfig",
    "FULL_METHOD_PRESET",
    "METHOD_PRESETS",
    "ServerlessOrchestrator",
    "load_window_dataset",
    "partition_train_dataset_for_satellites",
    "create_client_dataloaders",
    "train_one_federated_round",
    "run_federated_training",
]
