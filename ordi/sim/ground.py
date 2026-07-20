"""Ground-segment compute profiles used after raw-data downlink."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundComputeProfile:
    name: str
    compute_rate_flops_per_s: float
    active_power_w: float


# NVIDIA specifies 1,979 TFLOP/s FP16 Tensor Core throughput with sparsity and
# up to 700 W for H100 SXM.  Dense throughput is half the sparse figure; the
# simulator then assumes 50% sustained utilization for end-to-end inference.
# https://www.nvidia.com/en-us/data-center/h100/
H100_SXM_FP16_SPARSE_PEAK_TFLOPS = 1_979.0
H100_SXM_FP16_DENSE_PEAK_TFLOPS = (
    H100_SXM_FP16_SPARSE_PEAK_TFLOPS / 2.0
)
H100_SUSTAINED_UTILIZATION = 0.50
H100_SXM_MAX_POWER_W = 700.0

H100_SXM_PROFILE = GroundComputeProfile(
    name="NVIDIA H100 SXM (dense FP16, 50% sustained)",
    compute_rate_flops_per_s=(
        H100_SXM_FP16_DENSE_PEAK_TFLOPS
        * H100_SUSTAINED_UTILIZATION
        * 1e12
    ),
    active_power_w=H100_SXM_MAX_POWER_W,
)


__all__ = ["GroundComputeProfile", "H100_SXM_PROFILE"]
