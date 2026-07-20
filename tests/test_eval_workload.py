from collections import defaultdict

from ordi.eval.experiments import (
    _E1_BUILD_KWARGS, _EVALUATION_GS, _intensify_one_area_burst,
)
from ordi.tasks.generator import generate_tasks
from ordi.tasks.profiles import PROFILES


def test_evaluation_uses_six_geographically_distributed_stations():
    assert {station[0] for station in _EVALUATION_GS} == {
        "fairbanks", "greenwich", "singapore",
        "nairobi", "hawaii", "punta_arenas",
    }


def test_evaluation_deadlines_and_request_rate_match_workload_design():
    assert _E1_BUILD_KWARGS["arrival_rate"] == 16.0
    assert _E1_BUILD_KWARGS["burst_probability"] == 0.5
    assert _E1_BUILD_KWARGS["burst_size_range"] == (2, 4)
    assert _E1_BUILD_KWARGS["burst_window_s"] == 60.0
    assert _E1_BUILD_KWARGS["intense_area_compute_multiplier"] == 16.0
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
