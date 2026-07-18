"""Small common policy contract."""
from typing import Protocol

class Algorithm(Protocol):
    name: str
    def schedule_epoch(self, epoch: int, t_sim_start: float, pending_tasks: list): ...
