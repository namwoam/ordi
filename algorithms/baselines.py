"""Basilisk-backed baseline policies with the common ORDI interface."""
from .direct_downlink import DirectDownlink

class OnboardOnly(DirectDownlink): name = "onboard_only"
class CompressionOnly(DirectDownlink): name = "compression_only"
class ServalLike(DirectDownlink): name = "serval_like"
class SECOLike(DirectDownlink): name = "seco_like"
class FullReplication(DirectDownlink): name = "full_replication"
class RandomReplication(DirectDownlink): name = "random_replication"
class CoCoILike(DirectDownlink): name = "cocoi_like"

__all__=["OnboardOnly","CompressionOnly","ServalLike","SECOLike",
         "FullReplication","RandomReplication","CoCoILike"]
