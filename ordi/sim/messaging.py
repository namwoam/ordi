"""Deterministic discrete-event messaging for decentralized ORDI execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import heapq
import math

from ordi.algorithms.schema import MessageEvent, WorkItem
from ordi.eval.validation import InvalidDecisionError


def _contact_key(contact):
    return (
        contact.source, contact.target, contact.opens, contact.closes,
        contact.rate_bps, contact.kind,
    )


@dataclass(frozen=True)
class ProtocolMessage:
    message_id: str
    kind: str
    sender: str
    receiver: str
    item: WorkItem
    bits: float
    group_id: int
    shard_id: int
    hop_count: int = 0
    max_hops: int = 12
    dedup_key: str = ""


@dataclass(frozen=True)
class ProtocolExecution:
    delivery_time: float
    events: tuple[MessageEvent, ...]
    message_count: int
    control_bits: float
    ground_bits: float


@dataclass(frozen=True)
class AdvertisementBatch:
    events: tuple[MessageEvent, ...] = ()
    message_count: int = 0
    control_bits: float = 0.0


@dataclass
class MessageSimulator:
    """Execute node messages on contact and compute resources.

    Ledgers persist across calls/epochs, just like the physical feasibility
    model. Each call is transactional: an invalid protocol does not reserve
    contacts, compute, or duplicate keys.
    """

    header_bits: float = 2048.0
    max_hops: int = 12
    max_split_depth: int = 3
    advertisement_bits: float = 1024.0
    contact_ready_at: dict[tuple, float] = field(default_factory=dict)
    contact_residual_bits: dict[tuple, float] = field(default_factory=dict)
    compute_ready_at: dict[str, float] = field(default_factory=dict)
    delivered_keys: set[tuple[str, str]] = field(default_factory=set)
    knowledge: dict[str, dict[str, tuple[object, float, float]]] = field(
        default_factory=dict
    )
    pending_advertisements: list[tuple[float, str, str, object, float]] = field(
        default_factory=list
    )
    advertised_epochs: set[int] = field(default_factory=set)
    _next_id: int = 0

    def _message_id(self, trial_counter):
        value = f"msg-{trial_counter:08d}"
        return value, trial_counter + 1

    def seed_knowledge(self, observer, satellites, generated_at=0.0,
                       delivered_at=0.0):
        """Explicit bootstrap hook for controlled tests or real telemetry."""
        directory = self.knowledge.setdefault(observer, {})
        for sat_id, state in satellites.items():
            directory[sat_id] = (state, generated_at, delivered_at)

    def prepare_epoch(self, request):
        """Deliver old advertisements and broadcast current neighbor state.

        Decisions made in this epoch can only use entries delivered before
        ``request.sim_time``. Newly transmitted advertisements become visible
        in a later epoch after their contact-constrained arrival time.
        """
        events = []
        remaining = []
        for delivered_at, observer, sender, view, generated_at in (
                self.pending_advertisements):
            if delivered_at <= request.sim_time + 1e-9:
                self.knowledge.setdefault(observer, {})[sender] = (
                    view, generated_at, delivered_at
                )
            else:
                remaining.append(
                    (delivered_at, observer, sender, view, generated_at)
                )
        self.pending_advertisements = remaining

        if request.epoch in self.advertised_epochs:
            return AdvertisementBatch()
        self.advertised_epochs.add(request.epoch)
        ready = self.contact_ready_at.copy()
        residual = self.contact_residual_bits.copy()
        counter = self._next_id
        scheduled_pairs = set()
        message_count = 0
        control_bits = 0.0
        horizon = request.sim_time + request.epoch_length
        for contact in sorted(request.contacts, key=lambda item: item.opens):
            pair = (contact.source, contact.target)
            if (pair in scheduled_pairs or contact.kind != "isl"
                    or contact.source not in request.satellites
                    or contact.target not in request.satellites
                    or contact.opens > horizon
                    or contact.closes < request.sim_time):
                continue
            state = request.satellites[contact.source]
            if not state.available:
                continue
            try:
                depart, finish = self._reserve_hop(
                    request, ready, residual, contact.source, contact.target,
                    self.advertisement_bits, request.sim_time,
                )
            except InvalidDecisionError:
                continue
            message_id, counter = self._message_id(counter)
            item = WorkItem(
                -1, -1, contact.target, contact.source, depth=0
            )
            message = ProtocolMessage(
                message_id, "state_advertisement", contact.source,
                contact.target, item, self.advertisement_bits, 0, 0,
                0, self.max_hops,
                f"state:{request.epoch}:{contact.source}:{contact.target}",
            )
            self._record(
                events, depart, "hop_sent", message,
                contact.source, contact.target,
            )
            self._record(
                events, finish, "hop_received", message,
                contact.target, contact.source,
            )
            self._record(
                events, finish, "delivered", message,
                contact.target, contact.source,
            )
            self.pending_advertisements.append((
                finish, contact.target, contact.source, state,
                request.sim_time,
            ))
            scheduled_pairs.add(pair)
            message_count += 1
            control_bits += self.advertisement_bits

        self.contact_ready_at = ready
        self.contact_residual_bits = residual
        self._next_id = counter
        return AdvertisementBatch(
            tuple(sorted(events, key=lambda event: event.time)),
            message_count, control_bits,
        )

    def local_view(self, request, observer):
        """Return only state known to ``observer``, preserving stale values."""
        known = {observer: request.satellites[observer]}
        ages = {observer: 0.0}
        for sat_id, (view, generated_at, delivered_at) in (
                self.knowledge.get(observer, {}).items()):
            if delivered_at <= request.sim_time + 1e-9:
                known[sat_id] = view
                ages[sat_id] = max(0.0, request.sim_time - generated_at)
        known_nodes = set(known) | set(request.ground_stations)
        local_contacts = tuple(
            contact for contact in request.contacts
            if (contact.source in known_nodes
                and contact.target in known_nodes)
        )
        local_opportunities = {
            node: tuple(
                neighbor for neighbor in request.opportunities.get(node, ())
                if neighbor in known_nodes
            )
            for node in known
        }
        return replace(
            request, satellites=known, contacts=local_contacts,
            opportunities=local_opportunities,
            state_age_s=ages, observer=observer,
        )

    @staticmethod
    def _record(events, time, event, message, node, peer=""):
        events.append(MessageEvent(
            time, event, message.message_id, message.kind, node, peer,
            message.bits, message.item.task_id, message.item.tile_id,
            message.group_id, message.shard_id,
        ))

    @staticmethod
    def _reserve_hop(request, ready, residual, source, target, bits, start):
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
            available = residual.get(key, capacity)
            if available + 1e-9 < bits:
                continue
            depart = max(start, contact.opens, ready.get(key, contact.opens))
            finish = depart + bits / max(contact.rate_bps, 1.0)
            if finish > contact.closes + 1e-9:
                continue
            residual[key] = available - bits
            ready[key] = finish
            return depart, finish
        raise InvalidDecisionError(
            f"message has no contact capacity for {source}->{target} "
            f"({bits:.3f} bits after t={start:.6f})"
        )

    def execute(self, request, task, tile, assignment):
        if not assignment.node_decisions:
            raise InvalidDecisionError(
                "self-organized assignment has no node decisions"
            )
        if any(local.item.depth > self.max_split_depth
               for local in assignment.node_decisions):
            raise InvalidDecisionError(
                f"protocol exceeds split-depth limit {self.max_split_depth}"
            )

        ready = self.contact_ready_at.copy()
        residual = self.contact_residual_bits.copy()
        compute_ready = self.compute_ready_at.copy()
        delivered_keys = self.delivered_keys.copy()
        counter = self._next_id
        events = []
        event_queue = []
        sequence = 0
        inboxes = {node: [] for node in request.satellites}
        completed = {}
        message_count = 0
        control_bits = 0.0
        ground_bits = 0.0

        terminal = [
            local for local in assignment.node_decisions
            if local.action == "execute_forward"
        ]
        if len(terminal) != len(assignment.helpers):
            raise InvalidDecisionError(
                "protocol terminal decisions do not match compute operations"
            )

        def push(time, event, message, path=()):
            nonlocal sequence
            heapq.heappush(
                event_queue, (time, sequence, event, message, tuple(path))
            )
            sequence += 1

        def transmit(message, path, start):
            nonlocal message_count, control_bits, ground_bits
            if not path:
                path = (message.sender,)
            if path[0] != message.sender or path[-1] != message.receiver:
                raise InvalidDecisionError(
                    f"message {message.message_id} path endpoints disagree "
                    "with sender/receiver"
                )
            hops = len(path) - 1
            if message.hop_count + hops > message.max_hops:
                raise InvalidDecisionError(
                    f"message {message.message_id} exceeds hop limit"
                )
            now = start
            self._record(events, now, "sent", message, message.sender,
                         message.receiver)
            for source, target in zip(path, path[1:]):
                depart, now = self._reserve_hop(
                    request, ready, residual, source, target,
                    message.bits, now,
                )
                self._record(events, depart, "hop_sent", message, source, target)
                self._record(events, now, "hop_received", message, target, source)
            push(now, "deliver", message, path)
            message_count += 1
            control_bits += self.header_bits * hops
            if path and path[-1] in request.ground_stations:
                ground_bits += message.bits

        # A source inbox receives the original job locally.
        root = assignment.node_decisions[0].item
        root_id, counter = self._message_id(counter)
        root_message = ProtocolMessage(
            root_id, "job_descriptor", task.source_sat, task.source_sat,
            root, 0.0,
            0, 0, 0, self.max_hops,
            f"job:{request.epoch}:{task.task_id}:{tile.tile_id}:root",
        )
        push(request.sim_time, "deliver", root_message, (task.source_sat,))

        while event_queue:
            now, _order, event, message, _path = heapq.heappop(event_queue)
            if now > task.deadline + 1e-9:
                raise InvalidDecisionError(
                    f"protocol message {message.message_id} misses deadline"
                )
            if event == "compute_complete":
                index = message.shard_id
                route_out, route_down = (
                    assignment.routes[index][1], assignment.routes[index][2]
                )
                combined = route_out
                if route_down:
                    combined += route_down[1:] if combined else route_down
                receiver = combined[-1] if combined else message.receiver
                result_id, counter = self._message_id(counter)
                result = ProtocolMessage(
                    result_id, "result_shard", message.receiver, receiver,
                    message.item,
                    tile.d_out_bits * message.item.output_fraction
                    + self.header_bits,
                    message.group_id, index, message.hop_count,
                    self.max_hops,
                    f"result:{request.epoch}:{task.task_id}:{tile.tile_id}:"
                    f"{message.group_id}:{index}:{receiver}",
                )
                transmit(result, combined, now)
                continue

            key = (message.receiver, message.dedup_key)
            if message.dedup_key and key in delivered_keys:
                self._record(
                    events, now, "duplicate_dropped", message,
                    message.receiver, message.sender,
                )
                continue
            if message.dedup_key:
                delivered_keys.add(key)
            inboxes.setdefault(message.receiver, []).append(message)
            self._record(
                events, now, "delivered", message,
                message.receiver, message.sender,
            )
            if message.kind == "job_descriptor":
                # Only after the source receives the root work item does its
                # local decision emit delegated/split/replicated child jobs.
                for index, (local, helper, route) in enumerate(zip(
                        terminal, assignment.helpers, assignment.routes)):
                    item = local.item
                    if item.current_node != helper:
                        raise InvalidDecisionError(
                            "terminal work item is not held by its compute helper"
                        )
                    route_in = route[0]
                    msg_id, counter = self._message_id(counter)
                    job_message = ProtocolMessage(
                        msg_id, "image_shard", task.source_sat, helper, item,
                        tile.d_in_bits * item.input_fraction
                        + self.header_bits,
                        item.group_id, index, 0, self.max_hops,
                        f"job:{request.epoch}:{task.task_id}:{tile.tile_id}:"
                        f"{item.group_id}:{index}:{helper}",
                    )
                    transmit(job_message, route_in, now)
                continue
            if message.kind == "image_shard":
                state = request.satellites.get(message.receiver)
                if state is None or not state.available:
                    raise InvalidDecisionError(
                        f"message delivered work to unavailable node "
                        f"{message.receiver!r}"
                    )
                compute_start = max(
                    now,
                    request.sim_time + state.queued_flops
                    / max(state.compute_rate, 1.0),
                    compute_ready.get(message.receiver, now),
                )
                compute_done = compute_start + (
                    tile.compute_ops * message.item.work_fraction
                    / max(state.compute_rate, 1.0)
                )
                compute_ready[message.receiver] = compute_done
                self._record(
                    events, compute_start, "compute_started", message,
                    message.receiver,
                )
                self._record(
                    events, compute_done, "compute_finished", message,
                    message.receiver,
                )
                push(compute_done, "compute_complete", message)
            elif message.kind == "result_shard":
                completed.setdefault(message.group_id, []).append(now)

        required = int(assignment.metadata.get("data_shards", 1))
        complete_groups = [
            max(times) for times in completed.values()
            if len(times) == required
        ]
        if not complete_groups:
            raise InvalidDecisionError(
                "protocol produced no complete reconstruction group"
            )
        delivery_time = min(complete_groups)
        if not math.isfinite(delivery_time):
            raise InvalidDecisionError("protocol has no finite delivery time")

        self.contact_ready_at = ready
        self.contact_residual_bits = residual
        self.compute_ready_at = compute_ready
        self.delivered_keys = delivered_keys
        self._next_id = counter
        return ProtocolExecution(
            delivery_time, tuple(sorted(events, key=lambda item: item.time)),
            message_count, control_bits, ground_bits,
        )


__all__ = [
    "AdvertisementBatch", "MessageSimulator", "ProtocolExecution",
    "ProtocolMessage",
]
