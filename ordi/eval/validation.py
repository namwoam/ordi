"""Model-side validation of scheduler decisions against shared resources."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from ordi.algorithms.schema import Assignment, Decision, EpochInput


class InvalidDecisionError(ValueError):
    """Raised when a policy submits a decision the modeled system cannot run."""


def _contact_key(contact):
    return (
        contact.source, contact.target, contact.opens, contact.closes,
        contact.rate_bps, contact.kind,
    )


@dataclass
class DecisionFeasibilityModel:
    """Reserve network and compute resources independently of the policy.

    Reservations persist across scheduling epochs because a decision may use a
    future contact window.  Validation is transactional: an invalid decision
    does not alter the model ledger.
    """

    contact_ready_at: dict[tuple, float] = field(default_factory=dict)
    contact_residual_bits: dict[tuple, float] = field(default_factory=dict)
    compute_ready_at: dict[str, float] = field(default_factory=dict)

    def _copy(self):
        return DecisionFeasibilityModel(
            self.contact_ready_at.copy(),
            self.contact_residual_bits.copy(),
            self.compute_ready_at.copy(),
        )

    def _reserve_hop(self, request, source, target, bits, start):
        candidates = sorted(
            (contact for contact in request.contacts
             if contact.source == source and contact.target == target),
            key=lambda contact: contact.opens,
        )
        for contact in candidates:
            key = _contact_key(contact)
            capacity = max(0.0, contact.closes - contact.opens) * max(
                contact.rate_bps, 0.0
            )
            residual = self.contact_residual_bits.get(key, capacity)
            if residual + 1e-9 < bits:
                continue
            depart = max(
                start, contact.opens,
                self.contact_ready_at.get(key, contact.opens),
            )
            finish = depart + bits / max(contact.rate_bps, 1.0)
            if finish > contact.closes + 1e-9:
                continue
            self.contact_residual_bits[key] = residual - bits
            self.contact_ready_at[key] = finish
            return finish
        raise InvalidDecisionError(
            f"no residual contact capacity for {source}->{target} "
            f"({bits:.3f} bits after t={start:.6f})"
        )

    def _reserve_path(self, request, path, bits, start):
        now = start
        for source, target in zip(path, path[1:]):
            now = self._reserve_hop(
                request, source, target, bits, now
            )
        return now

    def _reserve_compute(self, request, helper, work, start):
        state = request.satellites.get(helper)
        if state is None or not state.available:
            raise InvalidDecisionError(
                f"compute helper {helper!r} is unavailable"
            )
        initial_ready = request.sim_time + state.queued_flops / max(
            state.compute_rate, 1.0
        )
        compute_start = max(
            start, initial_ready, self.compute_ready_at.get(helper, start)
        )
        finish = compute_start + work / max(state.compute_rate, 1.0)
        self.compute_ready_at[helper] = finish
        return finish

    def validate_and_reserve(self, request: EpochInput, decision: Decision,
                             *, retime: bool = False):
        """Validate one decision and atomically reserve all accepted work."""
        trial = self._copy()
        accepted = []
        task_by_id = {task.task_id: task for task in request.tasks}
        tile_by_key = {
            (task.task_id, tile.tile_id): tile
            for task in request.tasks for tile in task.tiles
        }

        for assignment in decision.assignments:
            key = (assignment.task_id, assignment.tile_id)
            task = task_by_id.get(assignment.task_id)
            tile = tile_by_key.get(key)
            if task is None or tile is None:
                raise InvalidDecisionError(
                    f"assignment references unknown task/tile {key}"
                )

            finishes = []
            if assignment.downlink_only:
                path = tuple(assignment.metadata.get("path", ()))
                if len(path) < 2:
                    raise InvalidDecisionError(
                        f"direct-downlink assignment {key} has no route"
                    )
                bits = float(assignment.metadata.get(
                    "downlink_bits", tile.d_in_bits
                ))
                finishes.append(trial._reserve_path(
                    request, path, bits, request.sim_time
                ))
            else:
                if len(assignment.routes) != len(assignment.helpers):
                    raise InvalidDecisionError(
                        f"assignment {key} does not provide one route per helper"
                    )
                if len(assignment.aggregators) != len(assignment.helpers):
                    raise InvalidDecisionError(
                        f"assignment {key} does not provide one aggregator per helper"
                    )
                for index, helper in enumerate(assignment.helpers):
                    route_in, route_out, route_down = assignment.routes[index]
                    work_fraction = (
                        assignment.work_fractions[index]
                        if index < len(assignment.work_fractions) else 1.0
                    )
                    input_fraction = (
                        assignment.input_fractions[index]
                        if index < len(assignment.input_fractions) else 1.0
                    )
                    output_fraction = (
                        assignment.output_fractions[index]
                        if index < len(assignment.output_fractions) else 1.0
                    )
                    now = trial._reserve_path(
                        request, route_in,
                        tile.d_in_bits * input_fraction,
                        request.sim_time,
                    )
                    now = trial._reserve_compute(
                        request, helper,
                        tile.compute_ops * work_fraction, now,
                    )
                    now = trial._reserve_path(
                        request, route_out,
                        tile.d_out_bits * output_fraction, now,
                    )
                    now = trial._reserve_path(
                        request, route_down,
                        tile.d_out_bits * output_fraction, now,
                    )
                    finishes.append(now)

            if not finishes:
                raise InvalidDecisionError(
                    f"assignment {key} performs neither compute nor downlink"
                )
            if any(finish > task.deadline + 1e-9 for finish in finishes):
                raise InvalidDecisionError(
                    f"assignment {key} finishes after deadline "
                    f"t={task.deadline:.6f}"
                )

            required = int(assignment.metadata.get("data_shards", 1))
            required = max(1, required)
            if required > len(finishes):
                raise InvalidDecisionError(
                    f"assignment {key} requires {required} completions but "
                    f"provides {len(finishes)}"
                )
            modeled_finish = request.sim_time + float(
                assignment.metadata.get("latency", math.inf)
            )
            feasible_delivery = sorted(finishes)[required - 1]
            if not retime and feasible_delivery > modeled_finish + 1e-6:
                raise InvalidDecisionError(
                    f"assignment {key} reports delivery at "
                    f"t={modeled_finish:.6f}, but shared-resource execution "
                    f"cannot deliver before t={feasible_delivery:.6f}"
                )
            if retime:
                metadata = dict(assignment.metadata)
                metadata["latency"] = feasible_delivery - request.sim_time
                assignment = replace(assignment, metadata=metadata)
            accepted.append(assignment)

        self.contact_ready_at = trial.contact_ready_at
        self.contact_residual_bits = trial.contact_residual_bits
        self.compute_ready_at = trial.compute_ready_at
        return (Decision(decision.epoch, tuple(accepted))
                if retime else decision)

    def retime_and_reserve(self, request: EpochInput,
                           assignment: Assignment) -> Assignment:
        """Commit one policy placement and replace optimism with model time.

        Policies still choose helpers, replicas, and paths.  This method only
        serializes those submitted operations on the shared resource ledger;
        an assignment that cannot meet its deadline remains invalid.
        """
        decision = self.validate_and_reserve(
            request, Decision(request.epoch, (assignment,)), retime=True
        )
        return decision.assignments[0]
