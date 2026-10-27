"""Task 9: resource calendars with capacity, queueing and future reservations.

A :class:`Resource` keeps an explicit deterministic calendar of reservations on
a single integer-nanosecond timeline.  Acquisition never sleeps or blocks: the
earliest feasible start is computed from the existing reservations and the
window is reserved immediately, so callers can overlap independent stages and
look ahead at future windows (shadow reservations for the scheduler).
"""

from __future__ import annotations

from bisect import bisect_right, insort
from dataclasses import dataclass, replace
from enum import Enum

from strata_sim.contracts.schema import ContractError


class ResourceError(ContractError):
    """A resource invariant was violated."""


class CapacityError(ResourceError):
    """Capacity is negative, zero, or smaller than the requested amount."""


class ResourceKind(str, Enum):
    FPGA_COMPUTE = "fpga_compute"
    SSD_COMMAND = "ssd_command"
    SSD_DATA = "ssd_data"
    DDR = "ddr"
    PCIE = "pcie"
    NETWORK = "network"
    CPU = "cpu"


def _require_nonneg(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ResourceError(f"{name} must be a non-negative int, got {value!r}")


def _require_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ResourceError(f"{name} must be a positive int, got {value!r}")


@dataclass(frozen=True)
class Reservation:
    """A booked window; ``waited_ns`` is queueing delay before the window."""

    resource: str
    kind: ResourceKind
    task_id: str
    requested_ns: int
    start_ns: int
    end_ns: int
    amount: int = 1
    payload_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.resource, str) or not self.resource:
            raise ResourceError(f"resource must be a non-empty string, got {self.resource!r}")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ResourceError(f"task_id must be a non-empty string, got {self.task_id!r}")
        _require_nonneg("requested_ns", self.requested_ns)
        _require_nonneg("start_ns", self.start_ns)
        _require_nonneg("payload_bytes", self.payload_bytes)
        _require_positive("amount", self.amount)
        if self.end_ns < self.start_ns:
            raise ResourceError(
                f"reservation end {self.end_ns} is before start {self.start_ns}")
        if self.start_ns < self.requested_ns:
            raise ResourceError(
                f"reservation starts at {self.start_ns} before requested {self.requested_ns}")

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def waited_ns(self) -> int:
        return self.start_ns - self.requested_ns

    def to_dict(self) -> dict:
        return {
            "resource": self.resource,
            "kind": self.kind.value,
            "task_id": self.task_id,
            "requested_ns": self.requested_ns,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "amount": self.amount,
            "payload_bytes": self.payload_bytes,
            "waited_ns": self.waited_ns,
        }


class Resource:
    """Capacity plus a reservation calendar for one physical resource."""

    def __init__(self, name: str, kind: ResourceKind, capacity: int,
                 bytes_per_ns: int = 1) -> None:
        if not isinstance(name, str) or not name:
            raise ResourceError(f"name must be a non-empty string, got {name!r}")
        if not isinstance(kind, ResourceKind):
            raise ResourceError(f"kind must be a ResourceKind, got {kind!r}")
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise CapacityError(f"capacity must be an int, got {capacity!r}")
        if capacity < 1:
            raise CapacityError(f"capacity must be >= 1, got {capacity!r}")
        _require_positive("bytes_per_ns", bytes_per_ns)
        self.name = name
        self.kind = kind
        self.capacity = capacity
        self.bytes_per_ns = bytes_per_ns
        self._reservations: list[Reservation] = []
        # end-time index kept in sync with _reservations: lets capacity queries
        # skip reservations that ended before the requested window starts.
        self._by_end: list[Reservation] = []
        self._end_keys: list[int] = []
        self.bytes_moved = 0
        self.bytes_by_task: dict[str, int] = {}
        self.queue_count = 0
        self.wait_ns_total = 0
        self.wait_ns_max = 0

    @property
    def reservations(self) -> tuple[Reservation, ...]:
        return tuple(self._reservations)

    def usage_at(self, time_ns: int) -> int:
        _require_nonneg("time_ns", time_ns)
        idx = bisect_right(self._end_keys, time_ns)
        return sum(r.amount for r in self._by_end[idx:] if r.start_ns <= time_ns)

    def max_usage(self, start_ns: int, end_ns: int) -> int:
        _require_nonneg("start_ns", start_ns)
        _require_nonneg("end_ns", end_ns)
        if end_ns <= start_ns:
            return 0
        deltas: list[tuple[int, int]] = []
        start_idx = bisect_right(self._end_keys, start_ns)
        for r in self._by_end[start_idx:]:
            overlap_start = max(r.start_ns, start_ns)
            overlap_end = min(r.end_ns, end_ns)
            if overlap_start < overlap_end:
                deltas.append((overlap_start, r.amount))
                deltas.append((overlap_end, -r.amount))
        deltas.sort(key=lambda item: (item[0], item[1]))
        current = 0
        peak = 0
        for _, delta in deltas:
            current += delta
            peak = max(peak, current)
        return peak

    def fits(self, start_ns: int, end_ns: int, amount: int = 1) -> bool:
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
            raise CapacityError(f"amount must be a positive int, got {amount!r}")
        if amount > self.capacity:
            return False
        if end_ns <= start_ns:
            return True
        return self.max_usage(start_ns, end_ns) + amount <= self.capacity

    def earliest_slot(self, start_ns: int, duration_ns: int, amount: int = 1) -> int:
        _require_nonneg("start_ns", start_ns)
        _require_positive("duration_ns", duration_ns)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
            raise CapacityError(f"amount must be a positive int, got {amount!r}")
        if amount > self.capacity:
            raise CapacityError(
                f"amount {amount} exceeds {self.name} capacity {self.capacity}")
        candidates = sorted(
            {start_ns} | {r.end_ns for r in self._by_end[bisect_right(self._end_keys, start_ns):]})
        for candidate in candidates:
            if self.fits(candidate, candidate + duration_ns, amount):
                return candidate
        raise ResourceError(f"no feasible slot on {self.name}")

    def acquire(self, task_id: str, start_ns: int, duration_ns: int, amount: int = 1,
                payload_bytes: int = 0) -> Reservation:
        if not isinstance(task_id, str) or not task_id:
            raise ResourceError(f"task_id must be a non-empty string, got {task_id!r}")
        _require_nonneg("start_ns", start_ns)
        _require_positive("duration_ns", duration_ns)
        _require_nonneg("payload_bytes", payload_bytes)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
            raise CapacityError(f"amount must be a positive int, got {amount!r}")
        if amount > self.capacity:
            raise CapacityError(
                f"amount {amount} exceeds {self.name} capacity {self.capacity}")
        slot = self.earliest_slot(start_ns, duration_ns, amount)
        reservation = Reservation(
            resource=self.name, kind=self.kind, task_id=task_id, requested_ns=start_ns,
            start_ns=slot, end_ns=slot + duration_ns, amount=amount,
            payload_bytes=payload_bytes)
        self._insert(reservation)
        if slot > start_ns:
            self.queue_count += 1
            self.wait_ns_total += slot - start_ns
            self.wait_ns_max = max(self.wait_ns_max, slot - start_ns)
        if payload_bytes:
            self.record_bytes(task_id, payload_bytes)
        return reservation

    def reserve_window(self, task_id: str, start_ns: int, end_ns: int, amount: int = 1,
                       payload_bytes: int = 0) -> Reservation:
        if not isinstance(task_id, str) or not task_id:
            raise ResourceError(f"task_id must be a non-empty string, got {task_id!r}")
        _require_nonneg("start_ns", start_ns)
        _require_nonneg("end_ns", end_ns)
        _require_nonneg("payload_bytes", payload_bytes)
        if end_ns <= start_ns:
            raise ResourceError(
                f"reservation window [{start_ns}, {end_ns}) is not positive")
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
            raise CapacityError(f"amount must be a positive int, got {amount!r}")
        if amount > self.capacity:
            raise CapacityError(
                f"amount {amount} exceeds {self.name} capacity {self.capacity}")
        if not self.fits(start_ns, end_ns, amount):
            raise ResourceError(
                f"window [{start_ns}, {end_ns}) x{amount} overflows {self.name} "
                f"capacity {self.capacity}")
        reservation = Reservation(
            resource=self.name, kind=self.kind, task_id=task_id, requested_ns=start_ns,
            start_ns=start_ns, end_ns=end_ns, amount=amount, payload_bytes=payload_bytes)
        self._insert(reservation)
        if payload_bytes:
            self.record_bytes(task_id, payload_bytes)
        return reservation

    def record_bytes(self, task_id: str, nbytes: int) -> None:
        _require_nonneg("nbytes", nbytes)
        self.bytes_moved += nbytes
        self.bytes_by_task[task_id] = self.bytes_by_task.get(task_id, 0) + nbytes

    def release_task(self, task_id: str, now_ns: int) -> int:
        _require_nonneg("now_ns", now_ns)
        kept: list[Reservation] = []
        released = 0
        for r in self._reservations:
            if r.task_id != task_id or r.end_ns <= now_ns:
                kept.append(r)
            elif r.start_ns >= now_ns:
                released += 1
            else:
                kept.append(replace(r, end_ns=now_ns))
                released += 1
        if released == 0:
            return 0
        self._reservations = kept
        self._rebuild_end_index()
        return released

    def max_end_ns(self) -> int:
        return self._by_end[-1].end_ns if self._by_end else 0

    def _rebuild_end_index(self) -> None:
        self._reservations.sort(key=lambda r: (r.start_ns, r.end_ns))
        self._by_end = sorted(self._reservations, key=lambda r: (r.end_ns, r.start_ns))
        self._end_keys = [r.end_ns for r in self._by_end]

    def _insert(self, reservation: Reservation) -> None:
        insort(self._reservations, reservation, key=lambda r: (r.start_ns, r.end_ns))
        insort(self._by_end, reservation, key=lambda r: (r.end_ns, r.start_ns))
        insort(self._end_keys, reservation.end_ns)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "capacity": self.capacity,
            "bytes_per_ns": self.bytes_per_ns,
            "reservations": [r.to_dict() for r in self._reservations],
            "bytes_moved": self.bytes_moved,
            "queue_count": self.queue_count,
            "wait_ns_total": self.wait_ns_total,
            "wait_ns_max": self.wait_ns_max,
        }
