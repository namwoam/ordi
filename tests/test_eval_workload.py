from collections import defaultdict
from inspect import signature

from ordi.eval import experiments
from ordi.eval.experiments import (
    _E1_BUILD_KWARGS, _E1_FAULT_RATE, _E2_FAULT_RATES,
    _E4_REQUEST_RATES, _EVALUATION_GS, SIM_DURATION_S, SIM_ORBITS,
    _four_neighbor_walker_pairs,
    _intensify_one_area_burst, _intensify_repeated_area_bursts, run_E1,
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
    assert signature(run_E1).parameters["fault_rate"].default == 0.02
    assert _E1_BUILD_KWARGS["n_planes"] == 3
    assert _E1_BUILD_KWARGS["sats_per_plane"] == 12
    assert _E1_BUILD_KWARGS["orbit_altitude_km"] == 475.0
    assert _E1_BUILD_KWARGS["orbit_inclination_deg"] == 97.4
    assert _E1_BUILD_KWARGS["orbit_period_s"] == 5670.0
    assert _E1_BUILD_KWARGS["arrival_rate"] == 20.0
    assert _E1_BUILD_KWARGS["burst_probability"] == 0.6
    assert _E1_BUILD_KWARGS["burst_size_range"] == (3, 6)
    assert _E1_BUILD_KWARGS["burst_window_s"] == 60.0
    assert _E1_BUILD_KWARGS["intense_area_request_count"] == 10
    assert _E1_BUILD_KWARGS["intense_area_compute_multiplier"] == 1.0
    assert _E1_BUILD_KWARGS["intense_area_window_s"] == 30.0
    assert _E1_BUILD_KWARGS["intense_bursts_per_orbit"] == 1
    assert _E1_BUILD_KWARGS["ground_stations"] == DEFAULT_GROUND_STATIONS
    assert _E1_BUILD_KWARGS["min_elevation_deg"] == 10.0
    assert _E1_BUILD_KWARGS["isl_topology"] == "four_neighbor"
    assert _E1_BUILD_KWARGS["acquisition_mode"] == "groundtrack"
    assert _E1_BUILD_KWARGS["fov_range_km"] == 16.25
    assert _E1_BUILD_KWARGS["input_band_counts"] == {
        "ship": 3, "wildfire": 4, "change": 8, "cloud_filter": 8,
    }
    assert _E1_BUILD_KWARGS["background_compute_utilization"] == 0.15
    assert SIM_ORBITS == 5
    assert SIM_DURATION_S == 5 * _E1_BUILD_KWARGS["orbit_period_s"]
    rates = {
        _E1_BUILD_KWARGS["satellite_params_factory"](f"SAT_00_{i:02d}")
        .compute_rate_gflops
        for i in range(4)
    }
    assert len(rates) > 1
    assert all(3.0 <= rate <= 8.0 for rate in rates)
    reliability = _E1_BUILD_KWARGS["reliability_model_factory"]()
    assert reliability.default_isl_pi == 0.995
    assert reliability.default_downlink_pi == 0.98
    assert reliability.default_node_pi == 0.999
    assert {
        name: profile.deadline_median_s for name, profile in PROFILES.items()
    } == {
        "wildfire": 600.0,
        "ship": 900.0,
        "change": 1800.0,
        "cloud_filter": 5760.0,
    }


def _capture_experiment_configs(monkeypatch, runner):
    captured = []

    def fake_run(config_args, desc=""):
        captured.extend(config_args)
        return {
            job[0]: []
            for _build_kwargs, jobs, _seed in config_args
            for job in jobs
        }

    monkeypatch.setattr(experiments, "_run_configs_parallel", fake_run)
    monkeypatch.setattr(experiments, "_save_csv", lambda *args, **kwargs: None)
    runner(seed=7, n_seeds=1)
    return captured


def test_e2_changes_fault_rate_from_the_e1_setup(monkeypatch):
    configs = _capture_experiment_configs(monkeypatch, experiments.run_E2)

    assert len(configs) == 1
    build_kwargs, jobs, seed = configs[0]
    assert build_kwargs == _E1_BUILD_KWARGS
    assert seed == 7
    assert signature(experiments.run_E2).parameters["n_seeds"].default == 8
    assert {job[3][0][1] for job in jobs} == set(_E2_FAULT_RATES)
    assert _E1_FAULT_RATE in _E2_FAULT_RATES


def test_e3_changes_only_the_fault_scenario_from_the_e1_setup(monkeypatch):
    configs = _capture_experiment_configs(monkeypatch, experiments.run_E3)

    assert len(configs) == 1
    build_kwargs, jobs, seed = configs[0]
    assert build_kwargs == _E1_BUILD_KWARGS
    assert seed == 7
    assert signature(experiments.run_E3).parameters["n_seeds"].default == 8
    assert all(len(job) == 4 for job in jobs)
    assert {spec[0] for job in jobs for spec in job[3]} == {"plane_outage"}
    assert any(not job[3] for job in jobs)


def test_e4_changes_only_request_rate_from_the_e1_setup(monkeypatch):
    configs = _capture_experiment_configs(monkeypatch, experiments.run_E4)

    assert _E4_REQUEST_RATES == (20, 40, 60, 80)
    assert signature(experiments.run_E4).parameters["n_seeds"].default == 8
    assert len(configs) == 4
    for build_kwargs, jobs, seed in configs:
        changed = {
            key: value for key, value in build_kwargs.items()
            if _E1_BUILD_KWARGS[key] != value
        }
        assert set(changed) <= {"arrival_rate"}
        assert build_kwargs["n_planes"] == _E1_BUILD_KWARGS["n_planes"]
        assert build_kwargs["sats_per_plane"] == _E1_BUILD_KWARGS["sats_per_plane"]
        assert build_kwargs["arrival_rate"] in _E4_REQUEST_RATES
        assert all(job[3] == [("random_schedule", _E1_FAULT_RATE, 7)]
                   for job in jobs)
        assert seed == 7


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


def test_planetscope_band_counts_scale_only_inference_input():
    groundtrack = {
        "sat-a": [(0.0, 0.0, 0.0), (100.0, 1.0, 1.0)],
    }
    tasks = generate_tasks(
        ["sat-a"], 100.0,
        arrival_rate_per_orbit=100.0, orbit_period_s=100.0,
        deadline_lognorm_sigma=0.0, n_tiles_side=1, seed=4,
        sat_groundtrack=groundtrack, acquisition_mode="groundtrack",
        input_band_counts={name: 8 for name in PROFILES},
    )
    assert tasks
    for task in tasks:
        tile = task.tiles[0]
        normalized = tile.d_in_bits / tile.profile.d_in_bits
        expected = 8 / tile.profile.input_bands
        assert 0.9 * expected <= normalized <= 1.1 * expected


def test_repeated_hotspots_select_one_burst_per_orbit():
    tasks = generate_tasks(
        ["sat-a", "sat-b"], 200.0,
        arrival_rate_per_orbit=100.0, orbit_period_s=100.0,
        deadline_lognorm_sigma=0.0, n_tiles_side=1, seed=7,
        burst_probability=1.0, burst_size_range=(3, 3),
        burst_window_s=10.0,
    )
    selected = _intensify_repeated_area_bursts(
        tasks, 1.0, request_count=10, window_s=30.0,
        orbit_period_s=100.0, bursts_per_orbit=1,
    )
    assert len(selected) == 2
    assert all(count == 10 for _burst, count in selected)
    assert sum(bool(getattr(task, "intense_area", False)) for task in tasks) == 20
    assert len({task.task_id for task in tasks}) == len(tasks)


def test_four_neighbor_mesh_caps_each_walker_node_at_four():
    pairs = _four_neighbor_walker_pairs(3, 12)
    degree = defaultdict(int)
    for pair in pairs:
        for node in pair:
            degree[node] += 1
    assert len(degree) == 36
    assert set(degree.values()) == {4}
