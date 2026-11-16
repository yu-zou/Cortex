"""Task 10: byte-capacity versioned cache for PQ codes and raw vectors.

State machine per key ``(tenant, index_version, cluster_id, kind)``:
``ABSENT -> LOADING -> READY -> PINNED -> EVICTED``.

Guarantees:
- resident bytes (payload plus per-entry metadata) never exceed the configured
  capacity or the per-kind partition budget;
- concurrent misses on the same key coalesce into exactly one fill
  (singleflight);
- entries with a positive refcount are pinned: never evicted, never
  over-allocated, and a conflicting insert is rejected with ``CacheFull``;
- optional hot-cluster prefetch is bounded by a traffic cap, only uses trace
  entries already observed, and cannot evict pinned entries;
- optional byte-LFU eviction and heat/priority admission implement the
  placement strategy; when a DDR directory is attached every load, access,
  pin/unpin and eviction is mirrored into it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from strata_sim.cache.directory import DDRDirectory
from strata_sim.cache.policy import (
    ByteLFU,
    ByteLRU,
    CacheKey,
    CacheKind,
    HotClusterPrefetcher,
    PartitionPlan,
    coerce_kind,
)
from strata_sim.contracts.schema import ContractError, VersionState

_HITTABLE = frozenset({VersionState.READY, VersionState.PINNED})
_EVICTION_CHOICES = ("lru", "lfu")
_ADMISSION_CHOICES = ("always", "heat")


class CacheError(RuntimeError):
    """Base class for cache state violations."""


class CacheFull(CacheError):
    """The entry cannot fit without evicting an in-use (pinned) entry."""


class AdmissionRejected(CacheFull):
    """Heat/priority admission declined the placement without reordering victims."""


@dataclass
class CacheEntry:
    """One cache slot and its byte accounting."""

    key: CacheKey
    state: VersionState = VersionState.ABSENT
    payload: bytes | None = None
    payload_bytes: int = 0
    metadata_bytes: int = 0
    refcount: int = 0
    hits: int = 0

    @property
    def accounted_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes

    @property
    def is_pinned(self) -> bool:
        return self.refcount > 0 or self.state is VersionState.PINNED


@dataclass
class _InFlight:
    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


def _as_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, (bytearray, memoryview)):
        return bytes(payload)
    raise ContractError(f"cache payload must be bytes-like, got {type(payload).__name__}")


class VersionedCache:
    """Byte-capacity cache with explicit PQ/raw partitions and singleflight fills."""

    def __init__(
        self,
        capacity_bytes: int,
        ratios: Mapping[CacheKind | str, float] | None = None,
        metadata_overhead_bytes: int = 0,
        prefetch_traffic_cap_bytes: int = 0,
        lookback: int = 4,
        min_hits: int = 2,
        max_candidates: int = 4,
        eviction: str | ByteLRU | ByteLFU = "lru",
        admission: str = "always",
        heat_threshold: float = 0.0,
        directory: DDRDirectory | None = None,
    ) -> None:
        if not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
            raise ContractError(f"capacity_bytes must be a positive int, got {capacity_bytes!r}")
        if not isinstance(metadata_overhead_bytes, int) or metadata_overhead_bytes < 0:
            raise ContractError(
                f"metadata_overhead_bytes must be a non-negative int, got {metadata_overhead_bytes!r}"
            )
        if not isinstance(prefetch_traffic_cap_bytes, int) or prefetch_traffic_cap_bytes < 0:
            raise ContractError(
                f"prefetch_traffic_cap_bytes must be a non-negative int, "
                f"got {prefetch_traffic_cap_bytes!r}"
            )
        if isinstance(eviction, str):
            if eviction not in _EVICTION_CHOICES:
                raise ContractError(
                    f"eviction must be one of {_EVICTION_CHOICES}, got {eviction!r}")
        elif not isinstance(eviction, (ByteLRU, ByteLFU)):
            raise ContractError(f"unsupported eviction index {eviction!r}")
        if admission not in _ADMISSION_CHOICES:
            raise ContractError(
                f"admission must be one of {_ADMISSION_CHOICES}, got {admission!r}")
        if not isinstance(heat_threshold, (int, float)) or heat_threshold < 0:
            raise ContractError(f"heat_threshold must be non-negative, got {heat_threshold!r}")
        if directory is not None and not isinstance(directory, DDRDirectory):
            raise ContractError(f"directory must be a DDRDirectory, got {directory!r}")
        self.capacity_bytes = capacity_bytes
        self.metadata_overhead_bytes = metadata_overhead_bytes
        self.prefetch_traffic_cap_bytes = prefetch_traffic_cap_bytes
        self.eviction = eviction if isinstance(eviction, str) else type(eviction).__name__
        self.admission = admission
        self.heat_threshold = float(heat_threshold)
        self.directory = directory
        self._eviction_template = eviction
        self._plan = (
            PartitionPlan.from_ratios(capacity_bytes, ratios) if ratios is not None else None
        )
        if self._plan is not None and not isinstance(eviction, str):
            raise ContractError("partitioned caches need an eviction strategy name, not an index")
        self._shared_index = None if self._plan is not None else self._make_index(capacity_bytes)
        self._kind_index: dict[CacheKind, ByteLRU | ByteLFU] = {}
        if self._plan is not None:
            for kind in CacheKind:
                self._kind_index[kind] = self._make_index(self._plan.budget_for(kind))
        self._planner = HotClusterPrefetcher(
            lookback=lookback, min_hits=min_hits, max_candidates=max_candidates
        )
        self._lock = threading.RLock()
        self._entries: dict[CacheKey, CacheEntry] = {}
        self._inflight: dict[CacheKey, _InFlight] = {}
        self._prefetch_used = 0
        self._heat: dict[CacheKey, float] = {}
        self.fills = 0
        self.evictions = 0
        self.prefetch_admissions = 0
        self.admission_rejects = 0

    def _make_index(self, budget_bytes: int) -> ByteLRU | ByteLFU:
        if isinstance(self._eviction_template, str):
            if self._eviction_template == "lru":
                return ByteLRU(budget_bytes)
            return ByteLFU(budget_bytes)
        return self._eviction_template

    def budget_for(self, kind: CacheKind | str) -> int:
        """Byte budget for one partition; unpartitioned caches share the full capacity."""
        kind_value = coerce_kind(kind)
        if self._plan is None:
            return self.capacity_bytes
        return self._plan.budget_for(kind_value)

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return sum(
                entry.accounted_bytes
                for entry in self._entries.values()
                if entry.state in _HITTABLE
            )

    @property
    def prefetch_bytes(self) -> int:
        with self._lock:
            return self._prefetch_used

    def stats(self) -> dict:
        """Numeric accounting snapshot for evidence and debugging."""
        with self._lock:
            per_kind = {}
            for kind in CacheKind:
                entries = [
                    entry
                    for entry in self._entries.values()
                    if entry.key.kind is kind and entry.state in _HITTABLE
                ]
                per_kind[kind.value] = {
                    "budget_bytes": self.budget_for(kind),
                    "resident_bytes": sum(entry.accounted_bytes for entry in entries),
                    "entries": len(entries),
                }
            return {
                "capacity_bytes": self.capacity_bytes,
                "resident_bytes": self.resident_bytes,
                "metadata_overhead_bytes": self.metadata_overhead_bytes,
                "prefetch_bytes": self._prefetch_used,
                "prefetch_cap_bytes": self.prefetch_traffic_cap_bytes,
                "fills": self.fills,
                "evictions": self.evictions,
                "admission_rejects": self.admission_rejects,
                "eviction": self.eviction,
                "admission": self.admission,
                "total_entries": len(self._entries),
                "per_kind": per_kind,
            }

    def state(self, key: CacheKey) -> VersionState:
        """Current state of a key: ABSENT for keys never installed."""
        with self._lock:
            entry = self._entries.get(key)
            return VersionState.ABSENT if entry is None else entry.state

    def is_resident(self, key: CacheKey) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and entry.state in _HITTABLE

    def resident_keys(
        self,
        *,
        version: str | None = None,
        kind: CacheKind | str | None = None,
    ) -> tuple[CacheKey, ...]:
        """Resident keys, optionally filtered by index version and partition kind."""
        kind_value = None if kind is None else coerce_kind(kind)
        with self._lock:
            return tuple(
                key
                for key, entry in self._entries.items()
                if entry.state in _HITTABLE
                and (version is None or key.index_version == version)
                and (kind_value is None or key.kind is kind_value)
            )

    def pinned_keys(self) -> tuple[CacheKey, ...]:
        with self._lock:
            return tuple(key for key, entry in self._entries.items() if entry.is_pinned)

    def get(self, key: CacheKey, now_ns: int = 0) -> bytes | None:
        """Return the payload only for a fully loaded, version-matching hit."""
        with self._lock:
            return self._peek_locked(key, now_ns)

    def heat(self, key: CacheKey) -> float:
        """Current admission heat/priority of a key (0.0 for never-seen keys)."""
        with self._lock:
            return self._heat.get(key, 0.0)

    def apply_heat(self, key: CacheKey, amount: float = 1.0, now_ns: int = 0) -> float:
        """Add heat to a key and mirror the access into the directory."""
        if not isinstance(amount, (int, float)) or amount < 0:
            raise ContractError(f"heat amount must be non-negative, got {amount!r}")
        with self._lock:
            self._heat[key] = self._heat.get(key, 0.0) + float(amount)
            if self.directory is not None:
                self.directory.touch(key, now_ns)
            return self._heat[key]

    def load(self, key: CacheKey, loader: Callable[[CacheKey], bytes],
             priority: float | None = None, now_ns: int = 0) -> bytes:
        """Fill a miss through ``loader``; concurrent misses on the key share one fill."""
        while True:
            with self._lock:
                hit = self._peek_locked(key, now_ns)
                if hit is not None:
                    return hit
                fill = self._inflight.get(key)
                leader = fill is None
                if leader:
                    fill = _InFlight()
                    self._inflight[key] = fill
                    self._entries[key] = CacheEntry(key=key, state=VersionState.LOADING)
            if not leader:
                assert fill is not None
                fill.event.wait()
                if fill.error is not None:
                    raise fill.error
                with self._lock:
                    hit = self._peek_locked(key, now_ns)
                if hit is not None:
                    return hit
                continue
            try:
                payload = _as_bytes(loader(key))
            except BaseException as exc:
                with self._lock:
                    fill.error = exc
                    self._cancel_loading_locked(key)
                    self._inflight.pop(key, None)
                fill.event.set()
                raise
            with self._lock:
                try:
                    self._install_locked(key, payload, priority=priority, now_ns=now_ns)
                except BaseException as exc:
                    fill.error = exc
                    self._cancel_loading_locked(key)
                    self._inflight.pop(key, None)
                    fill.event.set()
                    raise
                self._inflight.pop(key, None)
                self.fills += 1
                fill.event.set()
            return payload

    def get_or_load(self, key: CacheKey, loader: Callable[[CacheKey], bytes]) -> bytes:
        """Alias of :meth:`load` for miss-coalescing callers."""
        return self.load(key, loader)

    def put(self, key: CacheKey, payload: bytes | bytearray | memoryview,
            priority: float | None = None, now_ns: int = 0) -> CacheEntry:
        """Insert a fresh payload, evicting unpinned entries as needed."""
        data = _as_bytes(payload)
        with self._lock:
            return self._install_locked(key, data, priority=priority, now_ns=now_ns)

    def place(self, key: CacheKey, loader: Callable[[CacheKey], bytes] | None = None,
              payload: bytes | bytearray | memoryview | None = None, *,
              priority: float | None = None, now_ns: int = 0) -> bytes:
        """Free-space checked placement: evict-then-load under heat admission."""
        if payload is not None:
            data = _as_bytes(payload)
            self.put(key, data, priority=priority, now_ns=now_ns)
            return data
        if loader is None:
            raise ContractError("place requires either a payload or a loader")
        return self.load(key, loader, priority=priority, now_ns=now_ns)

    def preload(self, key: CacheKey, loader: Callable[[CacheKey], bytes], *,
                nprobe_trace: Sequence[int] | None = None, cursor: int | None = None,
                related_loader: Callable[[int], bytes] | None = None,
                priority: float | None = None,
                now_ns: int = 0) -> bytes:
        """Load an absent cluster from SSD then optionally prefetch related ones."""
        payload = self.load(key, loader, priority=priority, now_ns=now_ns)
        if nprobe_trace is not None and related_loader is not None:
            if cursor is None:
                cursor = len(nprobe_trace) - 1
            self.prefetch_clusters(nprobe_trace, cursor, base_key=key,
                                   loader=related_loader)
        return payload

    def acquire(self, key: CacheKey, now_ns: int = 0) -> bytes | None:
        """Pin a resident entry for compute; returns its payload, else None."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state not in _HITTABLE:
                return None
            entry.refcount += 1
            entry.state = VersionState.PINNED
            self._index_for(key.kind).touch(key)
            if self.directory is not None:
                self.directory.touch(key, now_ns)
                self.directory.acquire(key)
            return entry.payload

    def reserve(self, key: CacheKey) -> bytes | None:
        """Alias of :meth:`acquire` for reservation-style callers."""
        return self.acquire(key)

    def release(self, key: CacheKey) -> bool:
        """Drop one reservation; the entry becomes evictable at refcount zero."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state not in _HITTABLE:
                return False
            if entry.refcount > 0:
                entry.refcount -= 1
            if entry.refcount == 0 and entry.state is VersionState.PINNED:
                entry.state = VersionState.READY
            if self.directory is not None:
                self.directory.release(key)
            return True

    def pin(self, key: CacheKey) -> bool:
        """Explicitly pin a resident entry against eviction."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state not in _HITTABLE:
                return False
            entry.state = VersionState.PINNED
            if self.directory is not None:
                self.directory.pin(key)
            return True

    def unpin(self, key: CacheKey) -> bool:
        """Clear an explicit pin; refcounted entries stay unpinned-until-release."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state not in _HITTABLE:
                return False
            if entry.refcount == 0:
                entry.state = VersionState.READY
            if self.directory is not None:
                self.directory.unpin(key)
            return True

    def evict(self, key: CacheKey) -> bool:
        """Explicitly evict an unpinned resident entry; pinned entries raise."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state not in _HITTABLE:
                return False
            if entry.is_pinned:
                raise CacheError(f"cannot evict pinned entry {key!r}")
            self._retire_locked(key)
            self.evictions += 1
            return True

    def prefetch(self, key: CacheKey, payload: bytes | bytearray | memoryview) -> bool:
        """Admit prefetched bytes when under the traffic cap and never over budget."""
        data = _as_bytes(payload)
        cost = len(data)
        with self._lock:
            if self.is_resident(key):
                return False
            if cost > self.prefetch_traffic_cap_bytes - self._prefetch_used:
                return False
            try:
                self._install_locked(key, data)
            except CacheFull:
                return False
            self._prefetch_used += cost
            self.prefetch_admissions += 1
            return True

    def prefetch_clusters(
        self,
        trace: Sequence[int],
        cursor: int,
        base_key: CacheKey,
        loader: Callable[[int], bytes],
        *,
        kind: CacheKind | str | None = None,
    ) -> tuple[CacheKey, ...]:
        """Prefetch hot clusters observed at or before ``cursor`` within the traffic cap."""
        kind_value = base_key.kind if kind is None else coerce_kind(kind)
        with self._lock:
            resident = {
                key.cluster_id
                for key, entry in self._entries.items()
                if entry.state in _HITTABLE
                and key.tenant == base_key.tenant
                and key.index_version == base_key.index_version
                and key.kind is kind_value
            }
        candidates = self._planner.candidates(trace, cursor, resident_clusters=resident)
        admitted: list[CacheKey] = []
        for cluster_id in candidates:
            key = CacheKey(base_key.tenant, base_key.index_version, cluster_id, kind_value)
            if self.is_resident(key) or self.state(key) is VersionState.LOADING:
                continue
            if self.prefetch(key, loader(cluster_id)):
                admitted.append(key)
        return tuple(admitted)

    def _index_for(self, kind: CacheKind) -> ByteLRU | ByteLFU:
        if self._plan is None:
            assert self._shared_index is not None
            return self._shared_index
        return self._kind_index[kind]

    def _peek_locked(self, key: CacheKey, now_ns: int = 0) -> bytes | None:
        entry = self._entries.get(key)
        if entry is None or entry.state not in _HITTABLE:
            return None
        entry.hits += 1
        self._index_for(key.kind).touch(key)
        if self.directory is not None:
            self.directory.touch(key, now_ns)
        return entry.payload

    def _retire_locked(self, key: CacheKey, state: VersionState = VersionState.EVICTED) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        self._index_for(entry.key.kind).remove(key)
        entry.state = state
        entry.payload = None
        entry.payload_bytes = 0
        entry.metadata_bytes = 0
        if self.directory is not None:
            self.directory.record_remove(key, kind=entry.key.kind)

    def _cancel_loading_locked(self, key: CacheKey) -> None:
        entry = self._entries.get(key)
        if entry is not None and entry.state is VersionState.LOADING:
            self._retire_locked(key)

    def _admission_allows_locked(self, key: CacheKey, victims: list[CacheKey],
                                 priority: float | None) -> bool:
        if self.admission != "heat":
            return True
        incoming = self._heat.get(key, 0.0) if priority is None else float(priority)
        if incoming < self.heat_threshold:
            return False
        if victims:
            coldest = min(self._heat.get(victim, 0.0) for victim in victims)
            if incoming < coldest:
                return False
        return True

    def _install_locked(self, key: CacheKey, payload: bytes,
                        priority: float | None = None, now_ns: int = 0) -> CacheEntry:
        size = len(payload) + self.metadata_overhead_bytes
        budget = self.budget_for(key.kind)
        existing = self._entries.get(key)
        replace_bytes = (
            existing.accounted_bytes
            if existing is not None and existing.state in _HITTABLE
            else 0
        )
        if size > budget:
            raise CacheFull(
                f"entry {key!r} needs {size}B > {key.kind.value} budget {budget}B"
            )
        index = self._index_for(key.kind)
        victims = index.victims_for(
            size - replace_bytes, lambda victim: not self._entries[victim].is_pinned
        )
        if victims is None:
            raise CacheFull(
                f"cannot fit {size}B in {key.kind.value} budget {index.budget_bytes}B "
                f"without evicting pinned entries"
            )
        if not self._admission_allows_locked(key, victims, priority):
            self.admission_rejects += 1
            raise AdmissionRejected(
                f"heat admission rejected {key!r}: incoming "
                f"{self._heat.get(key, 0.0) if priority is None else priority} below "
                f"threshold {self.heat_threshold} or victim heat"
            )
        if replace_bytes:
            self._retire_locked(key)
        for victim in victims:
            self._retire_locked(victim)
            self.evictions += 1
        entry = CacheEntry(
            key=key,
            state=VersionState.READY,
            payload=payload,
            payload_bytes=len(payload),
            metadata_bytes=self.metadata_overhead_bytes,
        )
        self._entries[key] = entry
        index.add(key, size)
        if priority is not None:
            self._heat[key] = float(priority)
        if self.directory is not None:
            self.directory.record_load(key, kind=key.kind, payload_bytes=len(payload),
                                       now_ns=now_ns)
        return entry
