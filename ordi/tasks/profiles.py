"""
Per-tile compute and data profiles for four EO inference workloads.

Each profile gives:
  d_in_bits         : input tile size (bits)
  d_out_bits        : output result size (bits)
  compute_ops       : FLOPs required for inference on this tile
  utility           : base utility/priority weight
  deadline_median_s : per-type deadline median (seconds) used as the centre of
                      a log-normal distribution: wildfire 600 s, ship 900 s,
                      change 1800 s, and cloud filtering 5760 s (one orbit).

Each task is a 4096×4096 scene represented by a 4×4 grid of 1024×1024
logical tiles.  Input volume and inference work are area-scaled from the
original 128×128 MobileNetV2/YOLOv5-nano profiles.  Sparse detection products
stay compact; dense cloud/change masks scale with image area.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math


@dataclass(frozen=True)
class TileProfile:
    name: str
    d_in_bits: float         # bits
    d_out_bits: float        # bits (bounding boxes / mask / label + confidence)
    compute_ops: float       # FLOPs
    base_utility: float      # dimensionless priority weight
    deadline_median_s: float = 600.0  # per-type median deadline at reference scale
    input_bands: int = 3     # total channels represented by d_in_bits

    @property
    def d_in_bytes(self) -> float:
        return self.d_in_bits / 8

    @property
    def d_out_bytes(self) -> float:
        return self.d_out_bits / 8


# A 4×4 grid of these tiles represents one 4096×4096 scene.
_TILE_EDGE_PIX = 1024
_REFERENCE_TILE_EDGE_PIX = 128
_AREA_SCALE = (_TILE_EDGE_PIX / _REFERENCE_TILE_EDGE_PIX) ** 2  # 64×
_TILE_PIX = _TILE_EDGE_PIX * _TILE_EDGE_PIX * 3 * 8

# E1 treats this grid as a PlanetScope/SuperDove-class inference ROI rather
# than a complete framed scene. At native sampling it spans about 15.2 km per
# side and fits within the nominal SuperDove scene footprint.
PLANETSCOPE_NATIVE_GSD_M = 3.7
PLANETSCOPE_ROI_EDGE_PIX = 4096
PLANETSCOPE_TILE_EDGE_PIX = _TILE_EDGE_PIX
PLANETSCOPE_SCENE_KM = (32.5, 19.6)

PROFILES: Dict[str, TileProfile] = {
    "wildfire": TileProfile(
        name="wildfire",
        d_in_bits=_TILE_PIX,
        d_out_bits=1 * 8 * 1024,           # ~1 kB: class label + confidence + bounding box
        compute_ops=0.3e9 * _AREA_SCALE,    # area-scaled MobileNetV2
        base_utility=1.0,
        deadline_median_s=600.0,            # 10 min — urgent disaster alert
    ),
    "ship": TileProfile(
        name="ship",
        d_in_bits=_TILE_PIX,
        d_out_bits=5 * 8 * 1024,           # ~5 kB: multiple bounding boxes (YOLOv5-nano)
        compute_ops=0.9e9 * _AREA_SCALE,
        base_utility=0.8,
        deadline_median_s=900.0,            # 15 min — maritime alert
    ),
    "cloud_filter": TileProfile(
        name="cloud_filter",
        d_in_bits=_TILE_PIX,
        d_out_bits=50 * 8 * 1024 * _AREA_SCALE,  # dense pixel-wise mask
        compute_ops=1.2e9 * _AREA_SCALE,
        base_utility=0.5,
        deadline_median_s=5760.0,           # 96 min — finish within one orbit
    ),
    "change": TileProfile(
        name="change",
        d_in_bits=2 * _TILE_PIX,           # two co-registered tiles (before/after)
        d_out_bits=10 * 8 * 1024 * _AREA_SCALE,  # dense change map
        compute_ops=1.8e9 * _AREA_SCALE,    # area-scaled Siamese CNN
        base_utility=0.9,
        deadline_median_s=1800.0,           # 30 min — change analysis
        input_bands=6,                       # paired RGB observations
    ),
}

TASK_TYPES = list(PROFILES.keys())


def compute_time_seconds(profile: TileProfile, compute_rate_ops_per_s: float) -> float:
    """Wall-clock seconds to run inference on one tile."""
    return profile.compute_ops / max(compute_rate_ops_per_s, 1.0)
