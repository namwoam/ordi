from ordi.orbit.skyfield_contacts import (
    build_synthetic_walker,
    compute_contact_windows,
)


def test_isl_open_at_horizon_closes_at_last_sample():
    satellites = build_synthetic_walker(n_planes=1, sats_per_plane=2)
    horizon_end = 120.0

    contacts = compute_contact_windows(
        satellites,
        t_start_unix=0.0,
        t_end_unix=horizon_end,
        ground_stations=[],
        isl_max_range_km=float("inf"),
        dt_seconds=60.0,
    )

    isl = [contact for contact in contacts if contact.link_type == "isl"]
    assert len(isl) == 2
    assert all(contact.t_end <= horizon_end for contact in isl)
