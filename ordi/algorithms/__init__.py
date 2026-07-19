from .schema import (Algorithm, Assignment, ContactWindow, Decision, EpochInput,
                     ExperimentConfig, PolicyWeights, SatelliteView, snapshot)
from .ordi import ORDI
from .direct_downlink import DirectDownlink
from .onboard_only import OnboardOnly
from .compression_only import CompressionOnly
from .greedy_nonredundant import GreedyNonredundant
from .full_replication import FullReplication
from .random_replication import RandomReplication

ALL_ALGORITHMS = {
    cls.name: cls for cls in (
        ORDI, DirectDownlink, OnboardOnly, CompressionOnly,
        GreedyNonredundant, FullReplication, RandomReplication,
    )
}

__all__ = [
    "Algorithm", "Assignment", "ContactWindow", "Decision", "EpochInput",
    "ExperimentConfig", "PolicyWeights", "SatelliteView", "snapshot",
    "ORDI", "DirectDownlink", "OnboardOnly", "CompressionOnly",
    "GreedyNonredundant", "FullReplication", "RandomReplication",
    "ALL_ALGORITHMS",
]
