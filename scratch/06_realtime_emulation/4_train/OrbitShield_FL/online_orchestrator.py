"""Level 3 online co-simulation orchestrator for OrbitShield_FL."""

from __future__ import annotations

import os

from .config import FederatedConfig
from .ns3_online_bridge import Ns3OnlineRoundResult, generate_online_round_trace
from .serverless_orchestrator import ServerlessOrchestrator
from .topology_ns3 import convert_ns3_round_trace


class OnlineCosimOrchestrator(ServerlessOrchestrator):
    """Run federated training with in-the-loop ns-3 topology generation."""

    def __init__(self, config: FederatedConfig) -> None:
        super().__init__(config)
        self.online_trace_root = os.path.join(self.output_dir, "ns3_online_trace")
        os.makedirs(self.online_trace_root, exist_ok=True)
        self.online_trace_history: dict[int, str] = {}

    def _uses_ns3_backend(self) -> bool:
        """Report whether current backend is ns-3-based."""
        return True

    def _generate_online_round(self, round_idx: int) -> Ns3OnlineRoundResult:
        """Generate one fresh ns-3 round trace and cache its location."""
        result = generate_online_round_trace(
            binary_path=self.config.ns3_binary,
            trace_root_dir=self.online_trace_root,
            round_idx=round_idx,
            num_planes=self.config.num_planes,
            sats_per_plane=self.config.sats_per_plane,
            round_duration=self.config.ns3_round_duration,
            seed=self.config.seed,
            extra_args=[
                f"--contact-period={self.config.inter_plane_contact_period}",
                f"--contact-duration-rounds={self.config.inter_plane_contact_duration}",
                f"--intra-success-prob={self.config.intra_plane_success_prob}",
                f"--inter-success-prob={self.config.inter_plane_success_prob}",
                f"--inter-loss={self.config.packet_loss_prob}",
                f"--inter-delay={self.config.link_delay_mean}ms",
            ],
            force_regenerate=self.config.ns3_force_regenerate,
        )
        self.online_trace_history[round_idx] = str(result.trace_dir)
        return result

    def _get_topology_snapshot(self, round_idx: int) -> dict[str, object]:
        """Generate and return a fresh ns-3 topology snapshot for one round."""
        result = self._generate_online_round(round_idx)
        snapshot = convert_ns3_round_trace(result.round_trace)
        return snapshot

    def _write_outputs(self, final_test_metrics: dict[str, object], best_model_path: str) -> None:
        """Persist standard outputs and Level 3 online trace metadata."""
        super()._write_outputs(final_test_metrics, best_model_path)
        trace_index_path = os.path.join(self.output_dir, "ns3_online_trace_index.json")
        import json

        with open(trace_index_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "topology_backend": "ns3_online",
                    "trace_root": self.online_trace_root,
                    "round_trace_dirs": self.online_trace_history,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )


def run_online_federated_training(config: FederatedConfig) -> dict[str, object]:
    """Public API for Level 3 online ns-3 co-simulation training."""
    orchestrator = OnlineCosimOrchestrator(config)
    return orchestrator.run_federated_training()
