"""
Link and node reliability model.

p_kvia(t) = π_node_i(t) · π_path_ski(t) · π_path_ia(t) · π_down_a(t)

The independence assumption is an approximation; correlated failures
(orbital-plane outages, shared ground contacts) are tested in experiments.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math


# ── default reliability parameters ───────────────────────────────────────────
# ISL clear-sky, ground downlink clear-sky/adverse
DEFAULT_ISL_PI           = 0.97
DEFAULT_DOWNLINK_PI      = 0.92
DEFAULT_DOWNLINK_ADV_PI  = 0.70   # adverse weather / contact-miss scenario
DEFAULT_NODE_PI          = 0.98   # per-epoch node survival probability


@dataclass
class ReliabilityModel:
    """
    Holds per-link and per-node reliability parameters.
    Supports overrides for individual links (for fault injection).
    """
    default_isl_pi: float = DEFAULT_ISL_PI
    default_downlink_pi: float = DEFAULT_DOWNLINK_PI
    default_node_pi: float = DEFAULT_NODE_PI

    # Per-link overrides: (node_a, node_b) → probability
    _link_overrides: Dict[Tuple[str, str], float] = None
    # Per-node overrides: node_id → probability
    _node_overrides: Dict[str, float] = None
    # Per-aggregator downlink overrides: node_id → downlink probability
    # (adverse weather / degraded ground contact for specific aggregators)
    _downlink_overrides: Dict[str, float] = None
    _link_history: Dict[Tuple[str, str, int], float] = None
    _node_history: Dict[Tuple[str, int], float] = None
    _downlink_history: Dict[Tuple[str, int], float] = None

    def __post_init__(self):
        if self._link_overrides is None:
            self._link_overrides = {}
        if self._node_overrides is None:
            self._node_overrides = {}
        if self._downlink_overrides is None:
            self._downlink_overrides = {}
        if self._link_history is None:
            self._link_history = {}
        if self._node_history is None:
            self._node_history = {}
        if self._downlink_history is None:
            self._downlink_history = {}

    def link_pi(self, node_a: str, node_b: str, link_type: str = "isl",
                epoch: Optional[int] = None) -> float:
        """Probability that link (a→b) remains usable for a scheduled transfer."""
        key = (node_a, node_b)
        if epoch is not None and (*key, epoch) in self._link_history:
            return self._link_history[(*key, epoch)]
        if key in self._link_overrides:
            return self._link_overrides[key]
        if link_type == "downlink":
            return self.default_downlink_pi
        return self.default_isl_pi

    def node_pi(self, node_id: str, epoch: Optional[int] = None) -> float:
        """Survival probability of node during one epoch."""
        if epoch is not None and (node_id, epoch) in self._node_history:
            return self._node_history[(node_id, epoch)]
        if node_id in self._node_overrides:
            return self._node_overrides[node_id]
        return self.default_node_pi

    def downlink_pi(self, aggregator_id: str,
                    epoch: Optional[int] = None) -> float:
        """Probability that the aggregator's downlink to ground succeeds.
        Falls back to the clear-sky default; an adverse-weather fault sets a
        per-aggregator override (e.g. DEFAULT_DOWNLINK_ADV_PI)."""
        if (epoch is not None
                and (aggregator_id, epoch) in self._downlink_history):
            return self._downlink_history[(aggregator_id, epoch)]
        if aggregator_id in self._downlink_overrides:
            return self._downlink_overrides[aggregator_id]
        return self.default_downlink_pi

    def record_link_pi(self, node_a: str, node_b: str, epoch: int,
                       probability: float) -> None:
        self._link_history[(node_a, node_b, int(epoch))] = probability

    def record_node_pi(self, node_id: str, epoch: int,
                       probability: float) -> None:
        self._node_history[(node_id, int(epoch))] = probability

    def record_downlink_pi(self, node_id: str, epoch: int,
                           probability: float) -> None:
        self._downlink_history[(node_id, int(epoch))] = probability

    def path_pi(self, path: list, link_type: str = "isl") -> float:
        """
        Product of link reliabilities along a multi-hop path.
        path: list of node IDs [src, hop1, hop2, ..., dst]
        """
        if len(path) < 2:
            return 1.0
        pi = 1.0
        for i in range(len(path) - 1):
            pi *= self.link_pi(path[i], path[i + 1], link_type)
        return pi

    def replica_success_prob(
        self,
        helper_id: str,
        source_id: str,
        aggregator_id: str,
        src_helper_path: Optional[list] = None,
        helper_agg_path: Optional[list] = None,
        downlink_pi: Optional[float] = None,
    ) -> float:
        """
        p_kvia = π_node_i · π_path_ski · π_path_ia · π_down_a

        If paths are not provided, uses single-hop approximation.
        """
        # Both the helper (does the compute) and the source (must emit the input
        # tile) have to survive; a dead source cannot be rescued by any helper.
        pi_node = self.node_pi(helper_id) * self.node_pi(source_id)

        if src_helper_path:
            pi_ski = self.path_pi(src_helper_path, "isl")
        else:
            pi_ski = self.link_pi(source_id, helper_id, "isl")

        if helper_agg_path:
            pi_ia = self.path_pi(helper_agg_path, "isl")
        else:
            pi_ia = self.link_pi(helper_id, aggregator_id, "isl")

        if downlink_pi is not None:
            pi_down = downlink_pi
        else:
            pi_down = self.downlink_pi(aggregator_id)

        return pi_node * pi_ski * pi_ia * pi_down

    def tile_delivery_prob(self, replica_probs: list, source_pi: float = 1.0) -> float:
        """
        z_kv = π_source · [1 - Π_replicas (1 - p_kvia / π_source)]

        Source survival is a shared single point of failure, factored once at
        the tile level instead of per replica (which would overcount redundancy).
        """
        if not replica_probs or source_pi <= 0.0:
            return 0.0
        fail_all = 1.0
        for p in replica_probs:
            pc = min(1.0, p / source_pi)
            fail_all *= (1.0 - pc)
        return source_pi * (1.0 - fail_all)

    # ── fault injection helpers ───────────────────────────────────────────────

    def set_link_pi(self, node_a: str, node_b: str, pi: float):
        self._link_overrides[(node_a, node_b)] = pi

    def set_node_pi(self, node_id: str, pi: float):
        self._node_overrides[node_id] = pi

    def set_downlink_pi(self, aggregator_id: str, pi: float):
        self._downlink_overrides[aggregator_id] = pi

    def disable_link(self, node_a: str, node_b: str):
        self._link_overrides[(node_a, node_b)] = 0.0

    def disable_node(self, node_id: str):
        self._node_overrides[node_id] = 0.0

    def reset_overrides(self):
        self._link_overrides.clear()
        self._node_overrides.clear()
        self._downlink_overrides.clear()
