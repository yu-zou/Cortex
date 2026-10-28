"""Task 9: single logical-clock discrete-event engine.

The engine owns exactly one integer-nanosecond clock.  Every event is a
contract ``DesEvent`` ordered by ``(time_ns, phase, sequence)``; the contract
itself rejects wall-clock events, and this engine additionally rejects any
event or reservation placed before the current time.  Time advances only by
processing events -- nothing sleeps and no asyncio loop drives the clock.

Resource work is expressed as calendar reservations, so future windows can be
booked ahead (scheduler shadow reservations) and independent stages overlap.
A transfer may be chunked; each byte is recorded exactly once on its resource.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Callable

from strata_sim.contracts.schema import (
    ClockKind,
    ContractError,
    DesEvent,
    EventPhase,
    event_sort_key,
)
from strata_sim.des.resources import (
    Reservation,
    Resource,
    ResourceError,
    ResourceKind,
)

EventCallback = Callable[["DesEngine", DesEvent], None]
ChunkCallback = Callable[["DesEngine", Reservation, int], None]


class DesError(ContractError):
    """A DES invariant was violated."""


class TimeTravelError(DesError):
    """An event, acquisition or clock move targets the past."""


@dataclass(frozen=True)
class Transfer:
    resource: str
    task_id: str
    bytes_total: int
    chunks: tuple[Reservation, ...]
    start_ns: int
    end_ns: int

    @property
    def bytes_transferred(self) -> int:
        return sum(chunk.payload_bytes for chunk in self.chunks)

    def to_dict(self) -> dict:
        return {
            "resource": self.resource,
            "task_id": self.task_id,
            "bytes_total": self.bytes_total,
            "bytes_transferred": self.bytes_transferred,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


class DesEngine:
    """Deterministic reservation-based discrete-event engine."""

    def __init__(self, seed: int = 0) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise DesError(f"seed must be a non-negative int, got {seed!r}")
        self.seed = seed
        self.rng = random.Random(seed)
        self.now_ns = 0
        self.finish_times: dict[str, int] = {}
        self.event_log: list[DesEvent] = []
        self._resources: dict[str, Resource] = {}
        self._heap: list[tuple] = []
        self._sequence = 0
        self._push_id = 0

    def add_resource(self, name: str, kind: ResourceKind, capacity: int,
                     bytes_per_ns: int = 1) -> Resource:
        if name in self._resources:
            raise ResourceError(f"resource {name!r} already exists")
        resource = Resource(name, kind, capacity, bytes_per_ns)
        self._resources[name] = resource
        return resource

    def resource(self, name: str) -> Resource:
        try:
            return self._resources[name]
        except KeyError:
            raise ResourceError(f"unknown resource {name!r}") from None

    @property
    def resource_names(self) -> tuple[str, ...]:
        return tuple(self._resources)

    def advance_to(self, time_ns: int) -> int:
        self._check_time(time_ns, "advance_to")
        self.now_ns = time_ns
        return self.now_ns

    def schedule(self, time_ns: int, phase: EventPhase, *, task_id: str | None = None,
                 callback: EventCallback | None = None,
                 clock_kind: ClockKind = ClockKind.DES_LOGICAL,
                 sequence: int | None = None) -> DesEvent:
        self._check_time(time_ns, "event time")
        if clock_kind is not ClockKind.DES_LOGICAL:
            raise DesError(
                f"DES events must use clock_kind=des_logical, not {clock_kind!r}")
        if not isinstance(phase, EventPhase):
            raise DesError(f"phase must be an EventPhase, got {phase!r}")
        if sequence is None:
            sequence = self._next_sequence()
        event = DesEvent(time_ns=time_ns, phase=phase, sequence=sequence,
                         clock_kind=ClockKind.DES_LOGICAL)
        self._push(event, task_id, callback)
        return event

    def schedule_event(self, event: DesEvent, *, task_id: str | None = None,
                       callback: EventCallback | None = None) -> DesEvent:
        if not isinstance(event, DesEvent):
            raise DesError(f"expected a DesEvent, got {event!r}")
        if event.clock_kind is not ClockKind.DES_LOGICAL:
            raise DesError(
                f"DES events must use clock_kind=des_logical, not {event.clock_kind!r}")
        self._check_time(event.time_ns, "event time")
        self._push(event, task_id, callback)
        return event

    def run(self, until_ns: int | None = None) -> dict[str, int]:
        if until_ns is not None:
            self._check_time(until_ns, "run limit")
        limit = until_ns
        while self._heap and (limit is None or self._heap[0][0][0] <= limit):
            _, _, event, task_id, callback = heapq.heappop(self._heap)
            self.now_ns = event.time_ns
            self.event_log.append(event)
            if task_id is not None and event.phase is EventPhase.COMPLETE:
                self._record_finish(task_id, event.time_ns)
            if callback is not None:
                callback(self, event)
        if limit is None:
            target = max((resource.max_end_ns() for resource in self._resources.values()),
                         default=self.now_ns)
            self.now_ns = max(self.now_ns, target)
            return dict(self.finish_times)
        self.now_ns = max(self.now_ns, limit)
        return {task: finish for task, finish in self.finish_times.items()
                if finish <= self.now_ns}

    def complete(self, task_id: str, at_ns: int | None = None) -> int:
        at = self.now_ns if at_ns is None else at_ns
        self._check_time(at, "completion time")
        self._record_finish(task_id, at)
        return at

    def acquire(self, resource: str, task_id: str, duration_ns: int, *,
                start_ns: int | None = None, amount: int = 1, payload_bytes: int = 0,
                on_complete: Callable[["DesEngine", Reservation], None] | None = None,
                ) -> Reservation:
        target = self.resource(resource)
        start = self.now_ns if start_ns is None else start_ns
        self._check_time(start, "acquire start")
        reservation = target.acquire(task_id, start, duration_ns, amount=amount,
                                     payload_bytes=payload_bytes)
        self._record_finish(task_id, reservation.end_ns)
        if on_complete is not None:
            def fire(engine: DesEngine, event: DesEvent,
                     reservation: Reservation = reservation) -> None:
                on_complete(engine, reservation)
            self.schedule(reservation.end_ns, EventPhase.COMPLETE, task_id=task_id,
                          callback=fire)
        return reservation

    def reserve(self, resource: str, task_id: str, start_ns: int, end_ns: int, *,
                amount: int = 1, payload_bytes: int = 0) -> Reservation:
        target = self.resource(resource)
        self._check_time(start_ns, "reservation start")
        self._check_time(end_ns, "reservation end")
        reservation = target.reserve_window(task_id, start_ns, end_ns, amount=amount,
                                            payload_bytes=payload_bytes)
        self._record_finish(task_id, reservation.end_ns)
        return reservation

    def transfer(self, resource: str, task_id: str, nbytes: int, *,
                 chunk_bytes: int | None = None, start_ns: int | None = None,
                 bytes_per_ns: int | None = None,
                 on_chunk: ChunkCallback | None = None) -> Transfer:
        target = self.resource(resource)
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise DesError(f"nbytes must be a non-negative int, got {nbytes!r}")
        rate = target.bytes_per_ns if bytes_per_ns is None else bytes_per_ns
        if not isinstance(rate, int) or isinstance(rate, bool) or rate < 1:
            raise DesError(f"bytes_per_ns must be a positive int, got {rate!r}")
        start = self.now_ns if start_ns is None else start_ns
        self._check_time(start, "transfer start")
        if nbytes == 0:
            return Transfer(resource=target.name, task_id=task_id, bytes_total=0,
                            chunks=(), start_ns=start, end_ns=start)
        size = nbytes if chunk_bytes is None else chunk_bytes
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise DesError(f"chunk_bytes must be a positive int, got {chunk_bytes!r}")
        chunks: list[Reservation] = []
        cursor = start
        index = 0
        remaining = nbytes
        while remaining > 0:
            payload = min(size, remaining)
            duration = -(-payload // rate)
            reservation = target.acquire(task_id, cursor, duration, payload_bytes=payload)
            chunks.append(reservation)
            self._record_finish(task_id, reservation.end_ns)
            if on_chunk is not None:
                def ready(engine: DesEngine, event: DesEvent,
                          reservation: Reservation = reservation, index: int = index,
                          on_chunk: ChunkCallback = on_chunk) -> None:
                    on_chunk(engine, reservation, index)
                self.schedule(reservation.end_ns, EventPhase.LOAD, task_id=task_id,
                              callback=ready)
            cursor = reservation.end_ns
            remaining -= payload
            index += 1
        transfer = Transfer(resource=target.name, task_id=task_id, bytes_total=nbytes,
                            chunks=tuple(chunks), start_ns=chunks[0].start_ns,
                            end_ns=chunks[-1].end_ns)
        if transfer.bytes_transferred != nbytes:
            raise DesError(
                f"transfer accounting mismatch: {transfer.bytes_transferred} != {nbytes}")
        return transfer

    def release(self, resource: str, task_id: str) -> int:
        released = self.resource(resource).release_task(task_id, self.now_ns)
        self._recompute_finish(task_id)
        return released

    def _check_time(self, time_ns: int, what: str) -> None:
        if not isinstance(time_ns, int) or isinstance(time_ns, bool):
            raise DesError(f"{what} must be an integer nanosecond time, got {time_ns!r}")
        if time_ns < 0:
            raise DesError(f"{what} must be non-negative, got {time_ns!r}")
        if time_ns < self.now_ns:
            raise TimeTravelError(
                f"{what} {time_ns}ns is before engine now {self.now_ns}ns")

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def _push(self, event: DesEvent, task_id: str | None,
              callback: EventCallback | None) -> None:
        self._push_id += 1
        heapq.heappush(self._heap,
                       (event_sort_key(event), self._push_id, event, task_id, callback))

    def _record_finish(self, task_id: str, time_ns: int) -> None:
        current = self.finish_times.get(task_id)
        if current is None or time_ns > current:
            self.finish_times[task_id] = time_ns

    def _recompute_finish(self, task_id: str) -> None:
        ends = [r.end_ns for resource in self._resources.values()
                for r in resource.reservations if r.task_id == task_id]
        if ends:
            self.finish_times[task_id] = max(ends)
        else:
            self.finish_times.pop(task_id, None)


__all__ = [
    "ChunkCallback",
    "DesEngine",
    "DesError",
    "EventCallback",
    "TimeTravelError",
    "Transfer",
]
