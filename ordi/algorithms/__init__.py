from .schema import (Algorithm, Assignment, ContactWindow, Decision, EpochInput,
                     ExperimentConfig, PolicyWeights, SatelliteView, snapshot)
from .ordi import ORDI
from .direct_downlink import DirectDownlink
from .onboard_only import OnboardOnly
from .seco_adapted import SECOAdapted
from .full_replication import FullReplication
from .random_replication import RandomReplication

ALL_ALGORITHMS = {
    cls.name: cls for cls in (
        ORDI, DirectDownlink, OnboardOnly, SECOAdapted,
        FullReplication, RandomReplication,
    )
}

__all__ = [
    "Algorithm", "Assignment", "ContactWindow", "Decision", "EpochInput",
    "ExperimentConfig", "PolicyWeights", "SatelliteView", "snapshot",
    "ORDI", "DirectDownlink", "OnboardOnly", "SECOAdapted",
    "FullReplication", "RandomReplication",
    "ALL_ALGORITHMS",
]
