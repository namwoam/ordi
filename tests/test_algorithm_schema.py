from types import SimpleNamespace
from ordi.algorithms import (ALL_ALGORITHMS, ContactWindow, Decision, EpochInput,
                             SatelliteView)
from ordi.eval.metrics import EpochMetrics


def test_focused_evaluation_suite_exposes_only_four_distinct_questions():
    from ordi.eval.experiments import ALL_EXPERIMENTS

    assert list(ALL_EXPERIMENTS) == ["E1", "E2", "E3", "E4"]


def test_all_algorithms_share_epoch_input_decision_contract():
    tile=SimpleNamespace(tile_id=0,n_replicas_max=2,d_in_bits=1000.0,
                         d_out_bits=100.0,compute_ops=1e6,utility=1.0)
    task=SimpleNamespace(task_id=1,source_sat="SAT_00_00",deadline=120.0,tiles=[tile])
    states={sid:SatelliteView(sid,True,1e9,9000,10000,25,0)
            for sid in ("SAT_00_00","SAT_01_00")}
    request=EpochInput(0,0.0,[task],states,
        {"SAT_00_00":("SAT_01_00","ground")},frozenset({"ground"}),(
            ContactWindow("SAT_00_00","SAT_01_00",0,60,1e6,"isl"),
            ContactWindow("SAT_00_00","ground",0,60,1e6,"downlink"),
            ContactWindow("SAT_01_00","ground",0,60,1e6,"downlink"),
        ))
    for cls in ALL_ALGORITHMS.values():
        result=cls().schedule(request)
        assert isinstance(result,Decision)
        assert result.epoch==request.epoch

def test_controls_encode_distinct_replication_rules():
    tile=SimpleNamespace(tile_id=0,n_replicas_max=2,d_in_bits=1000.0,
                         d_out_bits=100.0,compute_ops=1e6,utility=1.0)
    task=SimpleNamespace(task_id=1,source_sat="SAT_00_00",deadline=120.0,tiles=[tile])
    states={sid:SatelliteView(sid,True,1e9,9000,10000,25,0,reliability=0.98)
            for sid in ("SAT_00_00","SAT_01_00")}
    contacts=(ContactWindow("SAT_00_00","SAT_01_00",0,60,1e6,"isl",0.97),
              ContactWindow("SAT_01_00","SAT_00_00",0,60,1e6,"isl",0.97),
              ContactWindow("SAT_00_00","ground",0,60,1e6,"downlink",0.92),
              ContactWindow("SAT_01_00","ground",0,60,1e6,"downlink",0.92))
    request=EpochInput(0,0,[task],states,{},frozenset({"ground"}),contacts)
    from ordi.algorithms import DirectDownlink, FullReplication, SECOAdapted
    assert DirectDownlink().schedule(request).assignments[0].downlink_only
    seco = SECOAdapted(split_options=(1,)).schedule(request).assignments[0]
    assert seco.metadata["effective_replicas"] == 1.0
    replication = FullReplication()
    replication.messages.seed_knowledge(
        "SAT_00_00", states, generated_at=-60.0, delivered_at=0.0
    )
    assert len(replication.schedule(request).assignments[0].helpers)==2


def test_e1_report_can_exclude_ordi_utility_fields(tmp_path, monkeypatch):
    import ordi.eval.experiments as experiments

    monkeypatch.setattr(experiments, "RESULTS_DIR", str(tmp_path))
    experiments._save_csv(
        "E1_core",
        {"ORDI": [EpochMetrics(
            epoch=0, deadline_miss_ratio=0.2, isl_traffic_bits=100.0,
            delivered_utility=9.0, objective=8.0,
        )]},
        metric_keys=["deadline_miss_ratio", "isl_traffic_bits"],
    )

    header = (tmp_path / "E1_core.csv").read_text().splitlines()[0]
    assert header.split(",") == [
        "algorithm", "sample_count", "deadline_miss_ratio", "isl_traffic_bits",
        "deadline_miss_ratio_std", "isl_traffic_bits_std",
    ]
    assert "utility" not in header
    assert "objective" not in header
