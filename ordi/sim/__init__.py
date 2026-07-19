"""Simulation backends and protocol runtime.

Messaging exports are loaded lazily so importing ``ordi.sim.satellite`` does
not initialize the algorithm package and create an ORDI↔messaging cycle.
"""

__all__ = [
    "AdvertisementBatch", "MessageSimulator", "ProtocolExecution",
    "ProtocolMessage",
]


def __getattr__(name):
    if name in __all__:
        from . import messaging
        return getattr(messaging, name)
    raise AttributeError(name)
