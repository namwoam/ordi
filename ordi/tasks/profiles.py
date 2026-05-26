"""
Per-tile compute and data profiles for four EO inference workloads.

Each profile gives:
  d_in_bits   : input tile size (bits)
  d_out_bits  : output result size (bits)
  compute_ops : FLOPs required for inference on this tile
  utility     : base utility/priority weight

Values are derived from benchmarking MobileNetV2 and YOLOv5-nano on
typical 128×128-pixel tiles at uint8 resolution (3 channels).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math


@dataclass(frozen=True)
class TileProfile:
    name: str
    d_in_bits: float    # bits
    d_out_bits: float   # bits (bounding boxes / mask / label + confidence)
    compute_ops: float  # FLOPs
    base_utility: float # dimensionless priority weight

    @property
    def d_in_bytes(self) -> float:
        return self.d_in_bits / 8

    @property
    def d_out_bytes(self) -> float:
        return self.d_out_bits / 8


# Tile size: 128×128 pixels, 3 channels, uint8 → 49,152 bytes = 393,216 bits
_TILE_PIX = 128 * 128 * 3 * 8   # bits

PROFILES: Dict[str, TileProfile] = {
    "wildfire": TileProfile(
        name="wildfire",
        d_in_bits=_TILE_PIX,
        d_out_bits=1 * 8 * 1024,           # ~1 kB: class label + confidence + bounding box
        compute_ops=0.3e9,                  # 0.3 GFLOPs (MobileNetV2 classifier)
        base_utility=1.0,
    ),
    "ship": TileProfile(
        name="ship",
        d_in_bits=_TILE_PIX,
        d_out_bits=5 * 8 * 1024,           # ~5 kB: multiple bounding boxes (YOLOv5-nano)
        compute_ops=0.9e9,
        base_utility=0.8,
    ),
    "cloud_filter": TileProfile(
        name="cloud_filter",
        d_in_bits=_TILE_PIX,
        d_out_bits=50 * 8 * 1024,          # ~50 kB: pixel-wise mask (lightweight U-Net)
        compute_ops=1.2e9,
        base_utility=0.5,
    ),
    "change": TileProfile(
        name="change",
        d_in_bits=2 * _TILE_PIX,           # two co-registered tiles (before/after)
        d_out_bits=10 * 8 * 1024,          # ~10 kB: change map
        compute_ops=1.8e9,                  # Siamese CNN
        base_utility=0.9,
    ),
}

TASK_TYPES = list(PROFILES.keys())


def compute_time_seconds(profile: TileProfile, compute_rate_ops_per_s: float) -> float:
    """Wall-clock seconds to run inference on one tile."""
    return profile.compute_ops / max(compute_rate_ops_per_s, 1.0)
