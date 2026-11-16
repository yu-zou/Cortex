"""Task 13: DES-side scheduling of the host-CPU rerank phases.

The functional top-k is computed by :mod:`strata_sim.functional.rerank` and the
modeled durations from the injected profile are booked on the engine's own
resource calendars: SSD command/data, PCIe and DDR for the DDR path, and SSD
command/data plus CPU for the SSD-only path.  All time is logical integer
nanoseconds; nothing sleeps and no wall clock is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from strata_sim.cache.store import VersionedCache
from strata_sim.contracts.schema import EventPhase
from strata_sim.des.engine import DesEngine
from strata_sim.des.resources import Reservation, ResourceKind
from strata_sim.functional.ivfpq import DatasetRef
from strata_sim.functional.rerank import (
    RawVectorStore,
    RerankPath,
    RerankProfile,
    RerankResult,
    rerank,
)


@dataclass(frozen=True)
class RerankResources:
    """Engine resource names used by the rerank phases."""

    ssd_command: str = "ssd-cmd"
    ssd_data: str = "ssd-data"
    pcie: str = "pcie"
    ddr: str = "ddr"
    cpu: str = "cpu"


def add_rerank_resources(
    engine: DesEngine,
    *,
    ssd_bytes_per_ns: int = 1,
    pcie_bytes_per_ns: int = 1,
    ddr_bytes_per_ns: int = 1,
    cpu_capacity: int = 1,
    resources: RerankResources | None = None,
) -> RerankResources:
    """Install the five standard rerank resources on ``engine``."""
    names = RerankResources() if resources is None else resources
    engine.add_resource(names.ssd_command, ResourceKind.SSD_COMMAND, capacity=1)
    engine.add_resource(
        names.ssd_data, ResourceKind.SSD_DATA, capacity=1, bytes_per_ns=ssd_bytes_per_ns
    )
    engine.add_resource(names.pcie, ResourceKind.PCIE, capacity=1, bytes_per_ns=pcie_bytes_per_ns)
    engine.add_resource(names.ddr, ResourceKind.DDR, capacity=1, bytes_per_ns=ddr_bytes_per_ns)
    engine.add_resource(names.cpu, ResourceKind.CPU, capacity=cpu_capacity)
    return names


@dataclass(frozen=True)
class DesRerankOutcome:
    """Functional result plus the reservations it was scheduled with."""

    task_id: str
    path: RerankPath
    result: RerankResult
    reservations: tuple[Reservation, ...]
    finish_ns: int

    def bytes_by_resource(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for reservation in self.reservations:
            totals[reservation.resource] = (
                totals.get(reservation.resource, 0) + reservation.payload_bytes
            )
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "path": self.path.value,
            "finish_ns": self.finish_ns,
            "bytes_by_resource": self.bytes_by_resource(),
            "reservations": [reservation.to_dict() for reservation in self.reservations],
            "result": self.result.to_dict(),
        }


def _transfer_ns(nbytes: int, bytes_per_ns: int) -> int:
    if nbytes <= 0:
        return 0
    return -(-nbytes // bytes_per_ns)


def _book(
    engine: DesEngine,
    resource: str,
    task_id: str,
    duration_ns: int,
    *,
    start_ns: int,
    payload_bytes: int,
    booked: list[Reservation],
) -> int:
    if duration_ns <= 0:
        return start_ns
    reservation = engine.acquire(
        resource, task_id, duration_ns, start_ns=start_ns, payload_bytes=payload_bytes
    )
    booked.append(reservation)
    return reservation.end_ns


def schedule_rerank(
    engine: DesEngine,
    task_id: str,
    candidates: Any,
    query: Any,
    store: RawVectorStore | None,
    *,
    ref: DatasetRef | None,
    k: int,
    path: RerankPath | str,
    profile: RerankProfile,
    cache: VersionedCache | None = None,
    resources: RerankResources | None = None,
    start_ns: int | None = None,
) -> DesRerankOutcome:
    """Compute the exact rerank and book its phases on the engine calendars."""
    names = RerankResources() if resources is None else resources
    result = rerank(
        candidates,
        query,
        store,
        ref=ref,
        k=k,
        path=path,
        profile=profile,
        cache=cache,
        record_wall=False,
    )
    accounting = result.accounting
    cursor = engine.now_ns if start_ns is None else int(start_ns)
    booked: list[Reservation] = []

    if result.path is RerankPath.SSD:
        if accounting.raw_bytes_read:
            cursor = _book(
                engine,
                names.ssd_command,
                task_id,
                profile.ssd_command_ns,
                start_ns=cursor,
                payload_bytes=0,
                booked=booked,
            )
            cursor = _book(
                engine,
                names.ssd_data,
                task_id,
                _transfer_ns(accounting.raw_bytes_read, profile.ssd_bytes_per_ns),
                start_ns=cursor,
                payload_bytes=accounting.raw_bytes_read,
                booked=booked,
            )
    elif result.path is RerankPath.DDR:
        if accounting.cache_bytes_filled:
            cursor = _book(
                engine,
                names.ssd_command,
                task_id,
                profile.ssd_command_ns,
                start_ns=cursor,
                payload_bytes=0,
                booked=booked,
            )
            cursor = _book(
                engine,
                names.ssd_data,
                task_id,
                _transfer_ns(accounting.cache_bytes_filled, profile.ssd_bytes_per_ns),
                start_ns=cursor,
                payload_bytes=accounting.cache_bytes_filled,
                booked=booked,
            )
            cursor = _book(
                engine,
                names.pcie,
                task_id,
                _transfer_ns(accounting.cache_bytes_filled, profile.pcie_bytes_per_ns),
                start_ns=cursor,
                payload_bytes=accounting.cache_bytes_filled,
                booked=booked,
            )
            cursor = _book(
                engine,
                names.ddr,
                task_id,
                _transfer_ns(accounting.cache_bytes_filled, profile.ddr_bytes_per_ns),
                start_ns=cursor,
                payload_bytes=accounting.cache_bytes_filled,
                booked=booked,
            )
        cursor = _book(
            engine,
            names.ddr,
            task_id,
            _transfer_ns(accounting.ddr_bytes_read, profile.ddr_bytes_per_ns),
            start_ns=cursor,
            payload_bytes=accounting.ddr_bytes_read,
            booked=booked,
        )

    cursor = _book(
        engine,
        names.cpu,
        task_id,
        accounting.cpu_ns,
        start_ns=cursor,
        payload_bytes=0,
        booked=booked,
    )
    finish_ns = cursor
    engine.schedule(finish_ns, EventPhase.COMPLETE, task_id=task_id)
    return DesRerankOutcome(
        task_id=task_id,
        path=result.path,
        result=result,
        reservations=tuple(booked),
        finish_ns=finish_ns,
    )


__all__ = [
    "DesRerankOutcome",
    "RerankResources",
    "add_rerank_resources",
    "schedule_rerank",
]
