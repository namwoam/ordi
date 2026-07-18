from .schema import (Algorithm, Assignment, ContactWindow, Decision, EpochInput,
                     ExperimentConfig, PolicyWeights, SatelliteView, snapshot)
from .ordi import ORDI
from .direct_downlink import DirectDownlink
from .onboard_only import OnboardOnly
from .compression_only import CompressionOnly
from .serval_like import ServalLike
from .seco_like import SECOLike
from .full_replication import FullReplication
from .random_replication import RandomReplication
from .cocoi_like import CoCoILike
from .ilp_reference import ILPReference
from .replanning import ReplanEvent, ReplanMonitor

ALL_ALGORITHMS={cls.name:cls for cls in (ORDI,DirectDownlink,OnboardOnly,CompressionOnly,
    ServalLike,SECOLike,FullReplication,RandomReplication,CoCoILike)}
__all__=["Algorithm","Assignment","ContactWindow","Decision","EpochInput","ExperimentConfig","PolicyWeights","SatelliteView","snapshot",
    "ORDI","DirectDownlink","OnboardOnly","CompressionOnly","ServalLike","SECOLike",
    "FullReplication","RandomReplication","CoCoILike","ILPReference",
    "ReplanEvent","ReplanMonitor","ALL_ALGORITHMS"]
