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
    reliability: float = 0.99

    @classmethod
    def from_state(cls, sat_id, state):
        return cls(sat_id, bool(state.A_i), state.C_i, state.B_i,
                   state.params.battery_j, state.Theta_i, state.Q_i,
                   reliability=0.99)

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
    # Number of execution epochs, including the post-arrival drain period.
    # ``None`` retains the legacy caller-selected/default horizon.
    simulation_epochs: int | None = None
    isl_rate_bps: float = 200e6
    max_backups: int = 1
    # A receiving ORDI node chooses the number of compute shards per tile.
    # Redundancy, when selected, creates another complete shard group.
    ordi_split_options: tuple[int, ...] = (1, 2, 4)
    plane_disjoint_backup: bool = False
    # SECO-aligned processing model.  A captured image tile may be split into
    # this many parallel shards.  Spatial splitting duplicates a small halo at
    # every internal boundary, so total input and compute grow with the split.
    seco_split_options: tuple[int, ...] = (1, 2, 4)
    split_halo_fraction: float = 0.05
    # Recurring non-ORDI accelerator demand. The physical queue receives this
    # fraction of one epoch's capacity before each scheduling round.
    background_compute_utilization: float = 0.0
    # Physical-environment configuration.  These values are populated by the
    # experiment builder and passed unchanged to Basilisk so its eclipse/power
    # orbit is the same Walker shell used to generate contacts and acquisitions.
    n_planes: int = 6
    sats_per_plane: int = 6
    orbit_altitude_km: float = 550.0
    orbit_inclination_deg: float = 53.0
    min_elevation_deg: float = 25.0
    ground_stations: tuple[tuple[str, float, float], ...] = ()

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
    # Populated for a node-local policy view. Missing entries mean that the
    # observer has never received a state advertisement from that satellite.
    state_age_s: Mapping[str, float] = field(default_factory=dict)
    observer: str | None = None


@dataclass(frozen=True)
class WorkItem:
    """A unit of work delivered to a node with its intended destination."""
    task_id: int
    tile_id: int
    destination: str | tuple[str, ...]
    current_node: str
    work_fraction: float = 1.0
    input_fraction: float = 1.0
    output_fraction: float = 1.0
    group_id: int = 0
    depth: int = 0


@dataclass(frozen=True)
class NodeDecision:
    """A node-local action taken after receiving a WorkItem."""
    node: str
    action: str
    item: WorkItem
    delegates: tuple[WorkItem, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class MessageEvent:
    """One immutable event emitted by the discrete-event protocol runtime."""
    time: float
    event: str
    message_id: str
    kind: str
    node: str
    peer: str = ""
    bits: float = 0.0
    task_id: int = -1
    tile_id: int = -1
    group_id: int = 0
    shard_id: int = 0

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
    # Auditable decentralized protocol trace. The physical model validates
    # that terminal execute/send actions agree with the submitted placement.
    node_decisions: tuple[NodeDecision, ...] = ()
    message_events: tuple[MessageEvent, ...] = ()

@dataclass(frozen=True)
class Decision:
    epoch: int
    assignments: tuple[Assignment, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    message_events: tuple[MessageEvent, ...] = ()

class Algorithm(Protocol):
    name: str
    def schedule(self, request: EpochInput) -> Decision: ...

def snapshot(backend, epoch, sim_time, tasks, opportunities, ground_stations=(),
             contacts=(), epoch_length=60.0, weights=PolicyWeights()):
    states = {sid: SatelliteView.from_state(sid, state)
              for sid, state in backend.states.items()}
    return EpochInput(epoch, sim_time, tasks, states, opportunities,
                      frozenset(ground_stations), contacts, epoch_length, weights)
