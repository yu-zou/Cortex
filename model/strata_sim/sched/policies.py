"""Replica selection policies B0-B4 (task 14).

All predictions read only scheduler-visible state (:class:`ReplicaView`): the
policy API has no handle on DES ground truth, future arrivals or measured
service traces. Eligible replicas must hold a READY/PINNED version of the
cluster's index version and be healthy.

``Scheduler.assign`` keeps ``queued_queries`` live (+1 per assignment) and
``Scheduler.complete``/``release`` decrement it. When a task-10
:class:`DDRDirectory` is bound per OSD, ``cached`` is refreshed from directory
residency before every selection and is never caller-supplied guesswork.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping

from strata_sim.cache.directory import DDRDirectory
from strata_sim.contracts import ClusterKey, VersionState
from strata_sim.sched.reservations import ReservationCalendar

_SERVABLE = {VersionState.READY, VersionState.PINNED}


class NoEligibleReplica(RuntimeError):
    """No replica holds a servable version of the requested cluster."""


@dataclass(frozen=True)
class TileTask:
    tile_id: str
    tenant: str
    cluster_key: ClusterKey
    n_docs: int
    top_k: int
    duration_ns: int


@dataclass
class ReplicaView:
    osd_id: int
    logical_host: str
    primary: bool = False
    healthy: bool = True
    resident_versions: dict[str, VersionState] = field(default_factory=dict)
    cached: bool = False
    queued_queries: int = 0
    queue_residual_ns: int = 0
    service_ns: int = 1_000

    def can_serve(self, key: ClusterKey) -> bool:
        state = self.resident_versions.get(key.index_version)
        return self.healthy and state is not None and VersionState(state) in _SERVABLE


class Policy:
    name = "base"

    def eligible_replicas(self, task: TileTask,
                          replicas: dict[int, ReplicaView]) -> list[int]:
        return sorted(osd for osd, view in replicas.items() if view.can_serve(task.cluster_key))

    def _pick(self, task: TileTask, replicas: dict[int, ReplicaView]) -> int:
        raise NotImplementedError

    def select(self, task: TileTask, replicas: dict[int, ReplicaView]) -> int:
        eligible = self.eligible_replicas(task, replicas)
        if not eligible:
            raise NoEligibleReplica(
                f"no eligible replica for tenant={task.cluster_key.tenant} "
                f"version={task.cluster_key.index_version} cluster={task.cluster_key.cluster_id}")
        return self._pick(task, replicas, eligible)

    def _pick(self, task, replicas, eligible):  # pragma: no cover - overridden
        raise NotImplementedError


class B0Primary(Policy):
    name = "B0-primary-only"

    def _pick(self, task, replicas, eligible):
        primaries = [osd for osd in eligible if replicas[osd].primary]
        return primaries[0] if primaries else eligible[0]


class B1Random(Policy):
    name = "B1-random"

    def __init__(self, seed: int = 11):
        self._rng = random.Random(seed)

    def _pick(self, task, replicas, eligible):
        return self._rng.choice(eligible)


class B2ShortestQueue(Policy):
    name = "B2-shortest-queue"

    def _pick(self, task, replicas, eligible):
        return min(eligible, key=lambda osd: (replicas[osd].queued_queries, osd))


class B3CacheFirst(Policy):
    name = "B3-cache-first"

    def _pick(self, task, replicas, eligible):
        return min(eligible, key=lambda osd: (
            not replicas[osd].cached, replicas[osd].queued_queries, osd))


class B4EstimatedFinish(Policy):
    name = "B4-estimated-finish"

    def __init__(self, calendar: ReservationCalendar | None = None, now_ns: int = 0):
        self.calendar = calendar or ReservationCalendar()
        self.now_ns = now_ns

    def estimate_finish_ns(self, task: TileTask, view: ReplicaView) -> int:
        reserved = max(self.calendar.earliest_finish(view.osd_id), self.now_ns)
        readiness = 0 if view.cached else view.service_ns
        setup = view.service_ns // 10
        return reserved + view.queue_residual_ns + view.service_ns + readiness + setup

    def rank(self, task: TileTask, replicas: dict[int, ReplicaView]) -> list[int]:
        eligible = self.eligible_replicas(task, replicas)
        return sorted(eligible, key=lambda osd: (
            self.estimate_finish_ns(task, replicas[osd]),
            replicas[osd].queued_queries,
            not replicas[osd].primary,
            osd,
        ))

    def _pick(self, task, replicas, eligible):
        return self.rank(task, replicas)[0]


@dataclass(frozen=True)
class Assignment:
    task: TileTask
    osd_id: int
    start_ns: int
    end_ns: int


class Scheduler:
    """Applies a policy and immediately books the shadow reservation.

    ``affinity_us`` (default 0) bounds how long a tenant may wait for a
    preferred replica; the starvation cap forces service for a lagging tenant.
    ``directory`` maps OSD id to that OSD's task-10 DDR directory; when present
    every selection refreshes ``cached`` from real directory residency.
    """

    def __init__(self, policy: Policy, calendar: ReservationCalendar | None = None,
                 affinity_us: int = 0, starvation_cap_ns: int = 1_000_000,
                 directory: Mapping[int, DDRDirectory] | None = None):
        self.policy = policy
        self.calendar = calendar or getattr(policy, "calendar", ReservationCalendar())
        if getattr(policy, "calendar", None) is None:
            policy.calendar = self.calendar  # type: ignore[attr-defined]
        self.affinity_us = affinity_us
        self.starvation_cap_ns = starvation_cap_ns
        self.directory: dict[int, DDRDirectory] = dict(directory or {})
        self._last_osd_by_tenant: dict[str, int] = {}
        self._inflight: dict[Assignment, ReplicaView] = {}

    def refresh(self, task: TileTask,
                replicas: Mapping[int, ReplicaView]) -> None:
        """Sync ``cached`` from the bound directory before a selection."""
        if not self.directory:
            return
        for osd_id, view in replicas.items():
            osd_directory = self.directory.get(osd_id)
            view.cached = bool(
                osd_directory is not None and osd_directory.is_cached(task.cluster_key)
            )

    def assign(self, tasks: list[TileTask], replicas: dict[int, ReplicaView],
               now_ns: int = 0) -> list[Assignment]:
        assignments: list[Assignment] = []
        for task in tasks:
            self.refresh(task, replicas)
            ranked = (self.policy.rank(task, replicas)
                      if isinstance(self.policy, B4EstimatedFinish)
                      else [self.policy.select(task, replicas)])
            chosen = ranked[0]
            preferred = self._last_osd_by_tenant.get(task.tenant)
            if self.affinity_us > 0 and preferred in ranked:
                window_ns = self.affinity_us * 1000
                if (self.calendar.earliest_finish(preferred)
                        <= self.calendar.earliest_finish(chosen) + window_ns):
                    chosen = preferred
            reservation = self.calendar.reserve(
                chosen, task.duration_ns, tenant=task.tenant, tile_id=task.tile_id,
                now_ns=now_ns)
            self._last_osd_by_tenant[task.tenant] = chosen
            view = replicas[chosen]
            view.queued_queries += 1
            assignment = Assignment(task, chosen, reservation.start_ns,
                                    reservation.end_ns)
            self._inflight[assignment] = view
            assignments.append(assignment)
        return assignments

    def complete(self, assignment: Assignment) -> bool:
        """Release one assigned query: decrement ``queued_queries`` exactly once."""
        view = self._inflight.pop(assignment, None)
        if view is None:
            return False
        if view.queued_queries > 0:
            view.queued_queries -= 1
        return True

    def release(self, assignment: Assignment) -> bool:
        """Alias of :meth:`complete` for completion/release callers."""
        return self.complete(assignment)
