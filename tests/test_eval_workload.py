from ordi.eval.experiments import _E1_BUILD_KWARGS, _EVALUATION_GS
from ordi.tasks.profiles import PROFILES


def test_evaluation_uses_six_geographically_distributed_stations():
    assert {station[0] for station in _EVALUATION_GS} == {
        "fairbanks", "greenwich", "singapore",
        "nairobi", "hawaii", "punta_arenas",
    }


def test_evaluation_deadlines_and_request_rate_match_workload_design():
    assert _E1_BUILD_KWARGS["arrival_rate"] == 16.0
    assert {
        name: profile.deadline_median_s for name, profile in PROFILES.items()
    } == {
        "wildfire": 600.0,
        "ship": 900.0,
        "change": 1800.0,
        "cloud_filter": 5760.0,
    }
