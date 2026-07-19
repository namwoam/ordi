"""Shared Basilisk-facing schema for every evaluated scheduling policy."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

@dataclass(frozen=True)
class SatelliteView:
    sat_id: str
    available: bool
    compute_rate: float
    battery_j: float
    battery_capacity_j: float
    temperature_c: float
    queued_flops: float
    compute_power_w: float = 0.0
    comms_power_w: float = 0.0
    reliability: float = 0.99

    @classmethod
    def from_state(cls, sat_id, state):
        return cls(sat_id, bool(state.A_i), state.C_i, state.B_i,
                   state.params.battery_j, state.Theta_i, state.Q_i,
                   state.params.compute_power_w, state.params.comms_power_w)

@dataclass(frozen=True)
class ContactWindow:
    source: str
    target: str
    opens: float
    closes: float
    rate_bps: float
    kind: str
    reliability: float = 0.99

@dataclass(frozen=True)
class PolicyWeights:
    freshness: float = 0.002
    energy: float = 1e-5
    communication: float = 1e-12
    replication: float = 0.05

@dataclass
class ExperimentConfig:
    lambda_E: float = 1e-5
    lambda_C: float = 1e-12
    lambda_R: float = 0.05
    alpha: float = 0.002
    epoch_length: float = 60.0
    isl_rate_bps: float = 200e6
    max_backups: int = 1
    plane_disjoint_backup: bool = False
    # SECO-aligned processing model.  A captured image tile may be split into
    # this many parallel shards.  Spatial splitting duplicates a small halo at
    # every internal boundary, so total input and compute grow with the split.
    seco_split_options: tuple[int, ...] = (1, 2, 4)
    split_halo_fraction: float = 0.05
    battery_reserve_frac: float = 0.15

@dataclass(frozen=True)
class EpochInput:
    epoch: int
    sim_time: float
    tasks: Sequence[Any]
    satellites: Mapping[str, SatelliteView]
    opportunities: Mapping[str, Sequence[str]]
    ground_stations: frozenset[str] = frozenset()
    contacts: Sequence[ContactWindow] = ()
    epoch_length: float = 60.0
    weights: PolicyWeights = PolicyWeights()

@dataclass(frozen=True)
class Assignment:
    task_id: int
    tile_id: int
    source: str
    helpers: tuple[str, ...] = ()
    aggregators: tuple[str, ...] = ()
    downlink_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    routes: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = ()
    # Parallel partitions are not replicas: each entry says what fraction of
    # the original tile's compute/input/output is assigned to the matching
    # helper.  Empty tuples preserve the historical whole-tile replica model.
    work_fractions: tuple[float, ...] = ()
    input_fractions: tuple[float, ...] = ()
    output_fractions: tuple[float, ...] = ()

@dataclass(frozen=True)
class Decision:
    epoch: int
    assignments: tuple[Assignment, ...] = ()

class Algorithm(Protocol):
    name: str
    def schedule(self, request: EpochInput) -> Decision: ...

def snapshot(backend, epoch, sim_time, tasks, opportunities, ground_stations=(),
             contacts=(), epoch_length=60.0, weights=PolicyWeights()):
    states = {sid: SatelliteView.from_state(sid, state)
              for sid, state in backend.states.items()}
    return EpochInput(epoch, sim_time, tasks, states, opportunities,
                      frozenset(ground_stations), contacts, epoch_length, weights)
