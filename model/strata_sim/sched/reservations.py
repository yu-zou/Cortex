"""Shadow reservation calendar and tenant queues (task 14).

All times are integer nanoseconds on the DES logical clock. A reservation only
exists on the scheduler side: it models expected device availability and is
updated immediately after each assignment so a batch is never pushed onto one
node.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Reservation:
    node_id: int
    tenant: str
    tile_id: str
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass
class ReservationCalendar:
    """Per-node non-overlapping future windows."""

    _finish: dict[int, int] = field(default_factory=dict)
    _load: dict[int, int] = field(default_factory=dict)
    entries: list[Reservation] = field(default_factory=list)

    def earliest_finish(self, node_id: int) -> int:
        return self._finish.get(node_id, 0)

    def load_ns(self, node_id: int) -> int:
        return self._load.get(node_id, 0)

    def reserve(self, node_id: int, duration_ns: int, *, tenant: str, tile_id: str,
                now_ns: int = 0) -> Reservation:
        if duration_ns < 0:
            raise ValueError(f"duration_ns must be non-negative, got {duration_ns}")
        if now_ns < 0:
            raise ValueError(f"now_ns must be non-negative, got {now_ns}")
        start = max(now_ns, self.earliest_finish(node_id))
        end = start + duration_ns
        reservation = Reservation(node_id, tenant, tile_id, start, end)
        self._finish[node_id] = end
        self._load[node_id] = self.load_ns(node_id) + duration_ns
        self.entries.append(reservation)
        return reservation

    def snapshot(self) -> dict[int, int]:
        return dict(self._finish)


@dataclass
class TenantedQueue:
    """Deficit round robin across tenants with a starvation cap.

    The cap forces a lagging tenant to be served when its head request has
    waited beyond ``starvation_cap_ns`` in logical time. Waiting time is
    advanced explicitly by the caller via :meth:`advance`.
    """

    tenants: list[str]
    weights: dict[str, int] | None = None
    quantum: int = 1
    starvation_cap_ns: int = 1_000_000
    _queues: dict[str, deque] = field(init=False, default_factory=dict)
    _deficit: dict[str, int] = field(init=False, default_factory=dict)
    _last_served: dict[str, int] = field(init=False, default_factory=dict)
    _rr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.weights = self.weights or {t: 1 for t in self.tenants}
        for tenant in self.tenants:
            self._queues[tenant] = deque()
            self._deficit[tenant] = 0
            self._last_served[tenant] = 0

    def enqueue(self, tenant: str, item) -> None:
        if tenant not in self._queues:
            raise KeyError(f"unknown tenant {tenant!r}")
        self._queues[tenant].append(item)

    def pending(self, tenant: str) -> int:
        return len(self._queues[tenant])

    def total_pending(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def next_tenant(self) -> str:
        if self.total_pending() == 0:
            raise IndexError("no queued requests")
        n = len(self.tenants)
        for offset in range(n * 2):
            idx = (self._rr + offset) % n
            tenant = self.tenants[idx]
            if not self._queues[tenant]:
                continue
            self._deficit[tenant] += self.weights.get(tenant, 1)
            if self._deficit[tenant] >= self.quantum:
                self._deficit[tenant] -= self.quantum
                self._rr = (idx + 1) % n
                return tenant
        for tenant in self.tenants:
            if self._queues[tenant]:
                return tenant
        raise IndexError("no queued requests")

    def next_item(self):
        tenant = self.next_tenant()
        return tenant, self._queues[tenant].popleft()
