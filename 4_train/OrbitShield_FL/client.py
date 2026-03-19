"""Federated client abstraction for simulated satellites."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dsc_cbam_gru import DSC_CBAM_GRU

from .aggregators import clone_state_dict, subtract_state_dict


@dataclass
class ClientTrainResult:
    """Container for local client update results."""

    weights: OrderedDict[str, torch.Tensor]
    update: OrderedDict[str, torch.Tensor]
    average_loss: float
    sample_count: int


class FederatedClient:
    """Simulated LEO satellite federated client."""

    def __init__(
        self,
        client_id: str,
        plane_id: int,
        train_loader: DataLoader,
        sample_count: int,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        bidirectional: bool,
        dropout: float,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.plane_id = plane_id
        self.train_loader = train_loader
        self.sample_count = sample_count
        self.reputation = 1.0
        self.last_sync_round = 0
        self.participation_count = 0
        self.successful_sync_count = 0
        self.model = DSC_CBAM_GRU(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.device = device
        self.model.to(self.device)

    def get_weights(self) -> OrderedDict[str, torch.Tensor]:
        """Return the current model weights on CPU."""
        return clone_state_dict(self.model.state_dict())

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load model weights from a CPU state dict."""
        self.model.load_state_dict(weights)
        self.model.to(self.device)

    def local_train(
        self,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        max_local_batches: int | None = None,
    ) -> ClientTrainResult:
        """Run local optimization for one federated round."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        before = self.get_weights()

        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for _ in range(local_epochs):
            for batch_idx, (features, labels) in enumerate(self.train_loader, start=1):
                features = features.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                total_batches += 1
                if max_local_batches is not None and batch_idx >= max_local_batches:
                    break

        after = self.get_weights()
        self.participation_count += 1
        return ClientTrainResult(
            weights=after,
            update=subtract_state_dict(after, before),
            average_loss=total_loss / max(total_batches, 1),
            sample_count=self.sample_count,
        )

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        """Evaluate client model on a provided loader."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                total_loss += float(loss.item()) * features.size(0)
                correct += int((preds == labels).sum().item())
                total_samples += features.size(0)
        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": correct / max(total_samples, 1),
        }
