import pytest

from ordi.eval.stochastic_oracle import (
    build_multi_epoch_instance, compare_stochastic_oracle,
    enumerate_stochastic_actions, solve_stochastic_oracle,
)


def test_multi_epoch_instance_contains_matched_correlated_faults():
    instance = build_multi_epoch_instance(
        seed=1, n_sats=4, n_requests=4, n_epochs=2
    )

    assert len(instance.requests) == 2
    assert sum(len(request.tasks) for request in instance.requests) == 4
    assert any(scenario.node_failures for scenario in instance.scenarios)
    assert any(scenario.link_failures for scenario in instance.scenarios)
    assert any(
        scenario.node_failures and scenario.link_failures
        for scenario in instance.scenarios
    )
    assert sum(
        scenario.weight for scenario in instance.scenarios
    ) == pytest.approx(1.0)
    nominal = next(
        scenario for scenario in instance.scenarios
        if scenario.name == "nominal"
    )
    assert nominal.weight == pytest.approx(0.90)


def test_action_space_contains_primary_and_disjoint_backup_choices():
    instance = build_multi_epoch_instance(
        seed=2, n_sats=4, n_requests=4, n_epochs=2
    )
    actions, *_rest = enumerate_stochastic_actions(
        instance, primary_cap=3, backup_cap=2
    )

    replica_counts = {
        len(action.helpers)
        for choices in actions.values() for action in choices
    }
    assert 1 in replica_counts
    assert 2 in replica_counts


def test_stochastic_oracle_bounds_online_policy_actions():
    records = compare_stochastic_oracle(
        seed=3, n_sats=4, n_requests=4, n_epochs=2,
        primary_cap=3, backup_cap=2,
    )

    assert {record.algorithm for record in records} == {
        "ORDI", "seco_adapted",
    }
    for record in records:
        assert record.oracle_objective + 1e-9 >= record.policy_objective
        assert 0.0 <= record.optimality_gap <= 1.0
        assert 0.0 <= record.oracle_expected_miss_ratio <= 1.0
        assert record.search_nodes > 0


def test_stochastic_oracle_respects_bounded_action_count():
    instance = build_multi_epoch_instance(
        seed=4, n_sats=4, n_requests=4, n_epochs=2
    )
    result = solve_stochastic_oracle(
        instance, primary_cap=2, backup_cap=1
    )

    assert result.candidate_count <= 4 * 3
    assert len(result.assignments) <= 4
