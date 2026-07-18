from types import SimpleNamespace
from ordi.algorithms import (ALL_ALGORITHMS, ContactWindow, Decision, EpochInput,
                             SatelliteView)

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

def test_baselines_encode_distinct_replication_rules():
    tile=SimpleNamespace(tile_id=0,n_replicas_max=2,d_in_bits=1000.0,
                         d_out_bits=100.0,compute_ops=1e6,utility=1.0)
    task=SimpleNamespace(task_id=1,source_sat="SAT_00_00",deadline=120.0,tiles=[tile])
    states={sid:SatelliteView(sid,True,1e9,9000,10000,25,0,10,5,0.98)
            for sid in ("SAT_00_00","SAT_01_00")}
    contacts=(ContactWindow("SAT_00_00","SAT_01_00",0,60,1e6,"isl",0.97),
              ContactWindow("SAT_00_00","ground",0,60,1e6,"downlink",0.92),
              ContactWindow("SAT_01_00","ground",0,60,1e6,"downlink",0.92))
    request=EpochInput(0,0,[task],states,{},frozenset({"ground"}),contacts)
    from ordi.algorithms import DirectDownlink, FullReplication, SECOLike, CoCoILike
    assert DirectDownlink().schedule(request).assignments[0].downlink_only
    assert len(SECOLike().schedule(request).assignments[0].helpers)==1
    assert len(FullReplication().schedule(request).assignments[0].helpers)==2
    coded=CoCoILike().schedule(request).assignments[0].metadata
    assert coded["coding"]=="mds" and coded["data_shards"]==1
