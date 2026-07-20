import pytest

from ordi.eval.oracle import (
    build_reduced_instance, compare_reduced_oracle,
    enumerate_oracle_candidates, solve_reduced_oracle,
)


def test_reduced_oracle_enforces_declared_size_and_candidate_bounds():
    request = build_reduced_instance(seed=2, n_tiles=3)
    candidates, _tiles = enumerate_oracle_candidates(request, candidate_cap=4)

    assert len(candidates) == 3
    assert all(0 < len(items) <= 4 for items in candidates.values())
    with pytest.raises(ValueError, match="at most 2 tiles"):
        solve_reduced_oracle(request, candidate_cap=4, max_tiles=2)


def test_reduced_oracle_bounds_existing_unsplit_policies():
    records = compare_reduced_oracle(seed=3, n_tiles=3, candidate_cap=5)

    assert {record.algorithm for record in records} == {
        "ORDI", "seco_adapted",
    }
    for record in records:
        assert record.oracle_objective + 1e-9 >= record.policy_objective
        assert 0.0 <= record.optimality_gap <= 1.0
        assert record.oracle_deliveries >= record.policy_deliveries
        assert record.search_nodes > 0
