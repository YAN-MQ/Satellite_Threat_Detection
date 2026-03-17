"""Serverless orchestration for federated multi-satellite training."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.dsc_cbam_gru import DSC_CBAM_GRU

from .aggregators import (
    StateDict,
    clone_state_dict,
    compute_staleness,
    estimate_link_quality,
    scale_state_dict,
    state_dict_num_bytes,
    subtract_state_dict,
    weighted_average_state_dict,
    intra_plane_aggregate,
)
from .client import FederatedClient
from .config import FederatedConfig
from .gossip import inter_plane_gossip
from .metrics_fl import evaluate_global_model, make_eval_loader, summarize_round_metrics
from .partition import create_client_dataloaders, dump_partition_stats, load_window_dataset, partition_train_dataset_for_satellites
from .reputation import compute_score, update_reputation
from .topology import build_plane_assignments, generate_topology_snapshot


class ServerlessOrchestrator:
    """Coordinate decentralized federated training across orbital planes."""

    def __init__(self, config: FederatedConfig) -> None:
        self.config = config
        self.config.validate()
        self.device = torch.device(config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.dataset = load_window_dataset(config.data_dir)
        self.partition_map = partition_train_dataset_for_satellites(
            train_npz_path=os.path.join(config.data_dir, "train.npz"),
            num_clients=config.num_clients,
            mode=config.partition_mode,
            alpha=config.dirichlet_alpha,
            seed=config.seed,
        )
        dump_partition_stats(os.path.join(self.output_dir, "partition_stats.json"), self.partition_map)

        train_x, train_y = self.dataset["train"]
        self.client_loaders = create_client_dataloaders(train_x, train_y, self.partition_map, batch_size=config.batch_size)
        self.global_train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        self.val_loader = make_eval_loader(*self.dataset["val"], batch_size=config.batch_size)
        self.test_loader = make_eval_loader(*self.dataset["test"], batch_size=config.batch_size)

        self.plane_assignments = build_plane_assignments(config.num_clients, config.num_planes)
        self.clients = self._build_clients()
        self.centralized_client = FederatedClient(
            client_id="single_global",
            plane_id=0,
            train_loader=self.global_train_loader,
            sample_count=len(train_y),
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
            device=self.device,
        )
        self.global_model = self._create_model()
        self.global_weights = clone_state_dict(self.global_model.state_dict())
        self._maybe_load_initial_checkpoint()
        self.plane_model_cache: dict[int, StateDict] = {
            plane_id: clone_state_dict(self.global_weights) for plane_id in range(config.num_planes)
        }
        self.plane_last_sync: dict[int, int] = {plane_id: 0 for plane_id in range(config.num_planes)}
        self.round_history: list[dict[str, object]] = []
        self.reputation_history: dict[str, list[float]] = {client_id: [1.0] for client_id in self.clients}
        self.topology_history: list[dict[str, object]] = []

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def _maybe_load_initial_checkpoint(self) -> None:
        """Warm start federated training from an existing single-node checkpoint."""
        if not self.config.init_checkpoint:
            return
        checkpoint_path = self.config.init_checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
        if not os.path.exists(checkpoint_path):
            print(f"[federated] init checkpoint not found, training from scratch: {checkpoint_path}")
            return

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._set_global_weights(state_dict)
        self.centralized_client.set_weights(self.global_weights)
        for client in self.clients.values():
            client.set_weights(self.global_weights)
        print(f"[federated] initialized from checkpoint: {checkpoint_path}")

    def _create_model(self) -> DSC_CBAM_GRU:
        """Instantiate the shared base model."""
        return DSC_CBAM_GRU(
            input_dim=self.config.input_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout,
        ).to(self.device)

    def _build_clients(self) -> dict[str, FederatedClient]:
        """Create federated client objects for all satellites."""
        clients: dict[str, FederatedClient] = {}
        for client_id, loader in self.client_loaders.items():
            plane_id = self.plane_assignments[client_id]
            sample_count = int(self.partition_map[client_id]["sample_count"])
            clients[client_id] = FederatedClient(
                client_id=client_id,
                plane_id=plane_id,
                train_loader=loader,
                sample_count=sample_count,
                input_dim=self.config.input_dim,
                num_classes=self.config.num_classes,
                hidden_dim=self.config.hidden_dim,
                bidirectional=self.config.bidirectional,
                dropout=self.config.dropout,
                device=self.device,
            )
        return clients

    def _set_global_weights(self, weights: StateDict) -> None:
        """Update the orchestrator's global model weights."""
        self.global_weights = clone_state_dict(weights)
        self.global_model.load_state_dict(self.global_weights)
        self.global_model.to(self.device)

    def _evaluate_loss_from_weights(self, weights: StateDict) -> float:
        """Evaluate validation loss for a candidate model state dict."""
        model = self._create_model()
        model.load_state_dict(weights)
        metrics = evaluate_global_model(model, self.val_loader, self.device)
        return float(metrics["loss"])

    def _serialize_topology(self, topology: dict[str, object], round_idx: int) -> dict[str, object]:
        """Make the topology snapshot JSON serializable."""
        inter_links = {}
        for pair, state in topology["inter_plane_links"].items():
            inter_links[f"{pair[0]}-{pair[1]}"] = asdict(state)
        intra_links = {}
        for pair, state in topology["intra_plane_links"].items():
            intra_links[f"{pair[0]}-{pair[1]}"] = asdict(state)
        return {
            "round": round_idx,
            "plane_neighbors": topology["plane_neighbors"],
            "inter_plane_links": inter_links,
            "intra_plane_links": intra_links,
        }

    def train_one_federated_round(self, round_idx: int) -> dict[str, object]:
        """Execute one federated training round across the constellation."""
        effective_method = self.config.method
        if self.config.method == "full" and round_idx <= self.config.warmup_rounds:
            effective_method = "intra_only"

        if self.config.method == "single":
            self.centralized_client.set_weights(self.global_weights)
            result = self.centralized_client.local_train(
                local_epochs=self.config.local_epochs,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            self._set_global_weights(result.weights)
            val_metrics = evaluate_global_model(self.global_model, self.val_loader, self.device)
            test_metrics = evaluate_global_model(self.global_model, self.test_loader, self.device)
            round_metrics = summarize_round_metrics(
                round_idx=round_idx,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                communication_cost_mb=0.0,
                stale_update_ratio=0.0,
                link_failure_robustness=1.0,
            )
            round_metrics["plane_meta"] = {}
            round_metrics["plane_gossip_meta"] = {}
            self.round_history.append(round_metrics)
            return round_metrics

        topology = generate_topology_snapshot(
            current_round=round_idx,
            num_planes=self.config.num_planes,
            intra_plane_success_prob=self.config.intra_plane_success_prob,
            inter_plane_success_prob=self.config.inter_plane_success_prob,
            inter_plane_contact_period=self.config.inter_plane_contact_period,
            inter_plane_contact_duration=self.config.inter_plane_contact_duration,
            packet_loss_prob=self.config.packet_loss_prob,
            link_delay_mean=self.config.link_delay_mean,
            seed=self.config.seed,
        )
        self.topology_history.append(self._serialize_topology(topology, round_idx))

        client_payloads_by_plane: dict[int, list[dict[str, object]]] = defaultdict(list)
        total_attempted_uploads = 0
        failed_uploads = 0
        stale_contributors = 0
        communication_bytes = 0
        previous_global = clone_state_dict(self.global_weights)

        for client_id, client in self.clients.items():
            total_attempted_uploads += 1
            client.set_weights(self.plane_model_cache[client.plane_id])
            intra_link = topology["intra_plane_links"][(client.plane_id, client.plane_id)]
            if not intra_link.success:
                failed_uploads += 1
                continue

            result = client.local_train(
                local_epochs=self.config.local_epochs,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            staleness = compute_staleness(client.last_sync_round, round_idx)
            stale_contributors += int(staleness > 1)
            link_quality = estimate_link_quality(
                success_rate=self.config.intra_plane_success_prob,
                contact_duration=intra_link.contact_duration,
                delay=intra_link.delay,
                packet_loss=intra_link.packet_loss,
            )
            client_payloads_by_plane[client.plane_id].append(
                {
                    "client_id": client_id,
                    "weights": result.weights,
                    "update": result.update,
                    "loss": result.average_loss,
                    "sample_count": result.sample_count,
                    "reputation": client.reputation,
                    "last_sync_round": client.last_sync_round,
                    "link_quality": link_quality,
                }
            )
            communication_bytes += state_dict_num_bytes(result.weights)

        plane_models: dict[int, StateDict] = {}
        plane_meta: dict[int, dict[str, object]] = {}
        plane_sample_counts: dict[int, int] = {}

        for plane_id in range(self.config.num_planes):
            payloads = client_payloads_by_plane.get(plane_id, [])
            if payloads:
                if self.config.method == "fedavg":
                    agg_method = "fedavg"
                elif effective_method in {"intra_only", "intra_gossip"}:
                    agg_method = effective_method
                else:
                    agg_method = "full"
                plane_model, metadata = intra_plane_aggregate(
                    payloads,
                    current_round=round_idx,
                    lambda_s=self.config.lambda_s,
                    method=agg_method,
                )
                plane_models[plane_id] = plane_model
                plane_meta[plane_id] = metadata
                plane_sample_counts[plane_id] = int(sum(payload["sample_count"] for payload in payloads))
                self.plane_last_sync[plane_id] = round_idx
            else:
                plane_models[plane_id] = clone_state_dict(self.plane_model_cache[plane_id])
                plane_meta[plane_id] = {
                    "client_weights": {},
                    "stale_clients": 0,
                    "participant_count": 0,
                }
                plane_sample_counts[plane_id] = 0

        gossiped_models: dict[int, StateDict] = {}
        plane_gossip_meta: dict[int, dict[str, object]] = {}
        total_inter_attempts = 0
        failed_inter_links = 0

        available_plane_models = {}
        for plane_id, model in plane_models.items():
            available_plane_models[plane_id] = model

        for plane_id in range(self.config.num_planes):
            neighbors = topology["plane_neighbors"][plane_id]
            visible_models = {}
            visible_scores = {}
            for neighbor in neighbors:
                total_inter_attempts += 1
                pair = tuple(sorted((plane_id, neighbor)))
                link_state = topology["inter_plane_links"][pair]
                if link_state.success:
                    visible_models[neighbor] = plane_models[neighbor]
                    visible_scores[neighbor] = estimate_link_quality(
                        success_rate=self.config.inter_plane_success_prob,
                        contact_duration=link_state.contact_duration,
                        delay=link_state.delay,
                        packet_loss=link_state.packet_loss,
                    )
                    communication_bytes += state_dict_num_bytes(plane_models[neighbor])
                else:
                    failed_inter_links += 1

            gossiped_model, gossip_meta = inter_plane_gossip(
                plane_id=plane_id,
                self_model=plane_models[plane_id],
                plane_neighbors=neighbors,
                available_models=visible_models,
                available_scores=visible_scores,
                plane_model_cache=self.plane_model_cache,
                plane_staleness={
                    other_plane: compute_staleness(self.plane_last_sync.get(other_plane, 0), round_idx)
                    for other_plane in range(self.config.num_planes)
                },
                beta=self.config.beta,
                beta_floor=self.config.beta_floor,
                lambda_s=self.config.lambda_s,
                method=effective_method,
                rho=self.config.rho,
            )
            gossiped_models[plane_id] = gossiped_model
            plane_gossip_meta[plane_id] = gossip_meta
            self.plane_model_cache[plane_id] = clone_state_dict(gossiped_model)

        if self.config.method == "fedavg":
            merged_weights = weighted_average_state_dict(
                list(plane_models.values()),
                [max(plane_sample_counts[plane_id], 1) for plane_id in range(self.config.num_planes)],
            )
        else:
            merged_weights = weighted_average_state_dict(
                list(gossiped_models.values()),
                [max(plane_sample_counts[plane_id], 1) for plane_id in range(self.config.num_planes)],
            )
            if effective_method == "full":
                merged_weights = weighted_average_state_dict(
                    [previous_global, merged_weights],
                    [self.config.global_momentum, 1.0 - self.config.global_momentum],
                )

        self._set_global_weights(merged_weights)

        for client_id, client in self.clients.items():
            client.set_weights(self.plane_model_cache[client.plane_id])
            client.last_sync_round = round_idx
            client.successful_sync_count += 1

        if effective_method == "full":
            previous_loss = self._evaluate_loss_from_weights(previous_global)
            plane_average_updates: dict[int, StateDict] = {}
            for plane_id in range(self.config.num_planes):
                plane_average_updates[plane_id] = subtract_state_dict(plane_models[plane_id], previous_global)

            for plane_id, payloads in client_payloads_by_plane.items():
                for payload in payloads:
                    client_id = str(payload["client_id"])
                    client = self.clients[client_id]
                    candidate_loss = self._evaluate_loss_from_weights(payload["weights"])
                    stability = client.successful_sync_count / max(client.participation_count, 1)
                    score, _ = compute_score(
                        client_update=payload["update"],
                        plane_average_update=plane_average_updates[plane_id],
                        loss_before=previous_loss,
                        loss_after=candidate_loss,
                        stable_i=stability,
                        weights=self.config.stable_score_weights,
                    )
                    client.reputation = update_reputation(
                        client.reputation,
                        score=score,
                        mu=self.config.mu,
                        r_min=self.config.r_min,
                    )
        for client_id, client in self.clients.items():
            self.reputation_history[client_id].append(client.reputation)

        val_metrics = evaluate_global_model(self.global_model, self.val_loader, self.device)
        test_metrics = evaluate_global_model(self.global_model, self.test_loader, self.device)
        round_metrics = summarize_round_metrics(
            round_idx=round_idx,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            communication_cost_mb=communication_bytes / (1024.0 * 1024.0),
            stale_update_ratio=stale_contributors / max(total_attempted_uploads, 1),
            link_failure_robustness=1.0 - ((failed_uploads + failed_inter_links) / max(total_attempted_uploads + total_inter_attempts, 1)),
        )
        round_metrics["plane_meta"] = plane_meta
        round_metrics["plane_gossip_meta"] = plane_gossip_meta
        self.round_history.append(round_metrics)
        return round_metrics

    def run_federated_training(self) -> dict[str, object]:
        """Run the full federated training experiment."""
        best_val_acc = -1.0
        best_weights = clone_state_dict(self.global_weights)

        for round_idx in range(1, self.config.rounds + 1):
            metrics = self.train_one_federated_round(round_idx)
            print(
                f"Round {round_idx:02d} | "
                f"val_acc={metrics['val_accuracy']:.4f} "
                f"test_acc={metrics['test_accuracy']:.4f} "
                f"comm_mb={metrics['communication_cost_mb']:.4f}"
            )
            if metrics["val_accuracy"] > best_val_acc:
                best_val_acc = float(metrics["val_accuracy"])
                best_weights = clone_state_dict(self.global_weights)

        self._set_global_weights(best_weights)
        final_test_metrics = evaluate_global_model(self.global_model, self.test_loader, self.device)
        best_model_path = os.path.join(self.output_dir, "best_global_model.pt")
        torch.save(best_weights, best_model_path)
        self._write_outputs(final_test_metrics, best_model_path)
        return {
            "best_val_accuracy": best_val_acc,
            "final_test_metrics": final_test_metrics,
            "best_model_path": best_model_path,
        }

    def _write_outputs(self, final_test_metrics: dict[str, object], best_model_path: str) -> None:
        """Persist round metrics and experiment summaries."""
        os.makedirs(self.output_dir, exist_ok=True)

        metrics_path = os.path.join(self.output_dir, "round_metrics.csv")
        with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "round",
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "communication_cost_mb",
                "stale_update_ratio",
                "link_failure_robustness",
                "confusion_matrix",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.round_history:
                writer.writerow({key: row[key] for key in fieldnames})

        with open(os.path.join(self.output_dir, "reputation_history.json"), "w", encoding="utf-8") as handle:
            json.dump(self.reputation_history, handle, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "topology_history.json"), "w", encoding="utf-8") as handle:
            json.dump(self.topology_history, handle, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": asdict(self.config),
                    "device": str(self.device),
                    "best_model_path": best_model_path,
                    "final_round": self.round_history[-1] if self.round_history else {},
                    "final_test_metrics": final_test_metrics,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )


def train_one_federated_round(orchestrator: ServerlessOrchestrator, round_idx: int) -> dict[str, object]:
    """Convenience wrapper required by the public API."""
    return orchestrator.train_one_federated_round(round_idx)


def run_federated_training(config: FederatedConfig) -> dict[str, object]:
    """Public API for end-to-end federated training."""
    orchestrator = ServerlessOrchestrator(config)
    return orchestrator.run_federated_training()
