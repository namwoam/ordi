"""Contact graph facade."""
from ordi.orbit._contact_types import ContactEvent, DEFAULT_GROUND_STATIONS
from ordi.orbit.skyfield_contacts import build_synthetic_walker, compute_contact_windows, compute_sat_groundtracks

__all__=["ContactEvent","DEFAULT_GROUND_STATIONS","build_synthetic_walker","compute_contact_windows","compute_sat_groundtracks"]
