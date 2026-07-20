from collections import defaultdict
from inspect import signature

from ordi.eval.experiments import (
    _E1_BUILD_KWARGS, _EVALUATION_GS, _intensify_one_area_burst, run_E1,
)
from ordi.orbit.contacts import DEFAULT_GROUND_STATIONS
from ordi.tasks.generator import generate_tasks
from ordi.tasks.profiles import PROFILES


def test_evaluation_uses_six_geographically_distributed_stations():
    assert {station[0] for station in _EVALUATION_GS} == {
        "fairbanks", "greenwich", "singapore",
        "nairobi", "hawaii", "punta_arenas",
    }


def test_evaluation_deadlines_and_request_rate_match_workload_design():
    assert signature(run_E1).parameters["n_seeds"].default == 8
    assert signature(run_E1).parameters["fault_rate"].default == 0.25
    assert _E1_BUILD_KWARGS["arrival_rate"] == 20.0
    assert _E1_BUILD_KWARGS["burst_probability"] == 0.6
    assert _E1_BUILD_KWARGS["burst_size_range"] == (3, 6)
    assert _E1_BUILD_KWARGS["burst_window_s"] == 60.0
    assert _E1_BUILD_KWARGS["intense_area_request_count"] == 8
    assert _E1_BUILD_KWARGS["intense_area_compute_multiplier"] == 16.0
    assert _E1_BUILD_KWARGS["intense_area_window_s"] == 30.0
    assert _E1_BUILD_KWARGS["ground_stations"] == DEFAULT_GROUND_STATIONS
    assert _E1_BUILD_KWARGS["min_elevation_deg"] == 10.0
    params = _E1_BUILD_KWARGS["satellite_params_factory"]("sat-a")
    assert params.compute_rate_gflops == 5.0
    assert {
        name: profile.deadline_median_s for name, profile in PROFILES.items()
    } == {
        "wildfire": 600.0,
        "ship": 900.0,
        "change": 1800.0,
        "cloud_filter": 5760.0,
    }


def test_scene_is_4096_square_with_area_scaled_work():
    wildfire = PROFILES["wildfire"]
    assert wildfire.d_in_bits == 1024 * 1024 * 3 * 8
    assert wildfire.compute_ops == 0.3e9 * 64


def test_bursts_share_one_hot_source_and_task_type():
    tasks = generate_tasks(
        ["sat-a", "sat-b"], 100.0,
        arrival_rate_per_orbit=100.0, orbit_period_s=100.0,
        deadline_lognorm_sigma=0.0, n_tiles_side=1, seed=7,
        burst_probability=1.0, burst_size_range=(3, 3),
        burst_window_s=10.0,
    )
    clusters = defaultdict(list)
    for task in tasks:
        clusters[task.burst_id].append(task)
    assert any(len(cluster) == 3 for cluster in clusters.values())
    for cluster in clusters.values():
        assert len({task.source_sat for task in cluster}) == 1
        assert len({task.task_type for task in cluster}) == 1
        assert cluster[-1].release_time - cluster[0].release_time <= 10.0


def test_one_area_burst_is_compute_intensified():
    tasks = generate_tasks(
        ["sat-a", "sat-b"], 100.0,
        arrival_rate_per_orbit=100.0, orbit_period_s=100.0,
        deadline_lognorm_sigma=0.0, n_tiles_side=1, seed=7,
        burst_probability=1.0, burst_size_range=(3, 3),
        burst_window_s=10.0,
    )
    before = {
        (task.task_id, tile.tile_id): tile.compute_ops
        for task in tasks for tile in task.tiles
    }

    selected = _intensify_one_area_burst(tasks, 4.0)

    assert selected is not None
    selected_burst, selected_count = selected
    assert selected_count >= 2
    for task in tasks:
        for tile in task.tiles:
            factor = (
                tile.compute_ops
                / before[(task.task_id, tile.tile_id)]
            )
            assert factor == (4.0 if task.burst_id == selected_burst else 1.0)


def test_intense_area_expands_to_eight_requests_without_growing_data():
    tasks = generate_tasks(
        ["sat-a", "sat-b"], 100.0,
        arrival_rate_per_orbit=100.0, orbit_period_s=100.0,
        deadline_lognorm_sigma=0.0, n_tiles_side=1, seed=7,
        burst_probability=1.0, burst_size_range=(3, 3),
        burst_window_s=60.0,
    )
    original_data_sizes = {
        (tile.d_in_bits, tile.d_out_bits)
        for task in tasks for tile in task.tiles
    }

    selected_burst, selected_count = _intensify_one_area_burst(
        tasks, 16.0, request_count=8, window_s=30.0,
    )

    hotspot = [task for task in tasks if task.burst_id == selected_burst]
    assert selected_count == 8
    assert len(hotspot) == 8
    assert len({task.source_sat for task in hotspot}) == 1
    assert len({task.task_type for task in hotspot}) == 1
    assert hotspot[-1].release_time - hotspot[0].release_time == 30.0
    assert all(
        (tile.d_in_bits, tile.d_out_bits) in original_data_sizes
        for task in hotspot for tile in task.tiles
    )
