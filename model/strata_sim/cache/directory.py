"""Task 10: per-OSD DDR cluster directory.

Every OSD owns one ``DDRDirectory`` that answers, per cluster, which of the PQ
codes and raw vectors are resident, the resident byte counts, access statistics,
pin/refcount state and an entry revision. ``snapshot``/``delta_since`` are the
messages the task-11 heartbeat ships; ``directory_hash`` is integrity and
change detection only and is never consulted by :meth:`DDRDirectory.contains`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Callable

from strata_sim.cache.policy import CacheKey, CacheKind, coerce_kind
from strata_sim.contracts.schema import ClusterKey, ContractError


def _nonneg(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative int, got {value!r}")


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string, got {value!r}")


def _ident(key: ClusterKey | CacheKey) -> tuple[str, str, int]:
    return (key.tenant, key.index_version, key.cluster_id)


@dataclass
class DirectoryEntry:
    """One cluster's residency facts on one OSD."""

    tenant: str
    index_version: str
    cluster_id: int
    pq_resident: bool = False
    raw_resident: bool = False
    pq_bytes: int = 0
    raw_bytes: int = 0
    access_count: int = 0
    last_access_ns: int = 0
    pinned: bool = False
    refcount: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        _nonempty("tenant", self.tenant)
        _nonempty("index_version", self.index_version)
        _nonneg("cluster_id", self.cluster_id)
        _nonneg("pq_bytes", self.pq_bytes)
        _nonneg("raw_bytes", self.raw_bytes)
        _nonneg("access_count", self.access_count)
        _nonneg("last_access_ns", self.last_access_ns)
        _nonneg("refcount", self.refcount)
        if not isinstance(self.version, int) or self.version < 1:
            raise ContractError(f"version must be a positive int, got {self.version!r}")

    @property
    def ident(self) -> tuple[str, str, int]:
        return (self.tenant, self.index_version, self.cluster_id)

    def cache_key(self, kind: CacheKind | str = CacheKind.PQ) -> CacheKey:
        return CacheKey(self.tenant, self.index_version, self.cluster_id, coerce_kind(kind))

    def resident(self, kind: CacheKind | str | None = None) -> bool:
        if kind is None:
            return self.pq_resident or self.raw_resident
        return self.pq_resident if coerce_kind(kind) is CacheKind.PQ else self.raw_resident

    def bytes_for(self, kind: CacheKind | str) -> int:
        return self.pq_bytes if coerce_kind(kind) is CacheKind.PQ else self.raw_bytes

    @property
    def total_bytes(self) -> int:
        return self.pq_bytes + self.raw_bytes

    @property
    def is_pinned(self) -> bool:
        return self.pinned or self.refcount > 0

    @property
    def is_evictable(self) -> bool:
        return not self.is_pinned

    def to_dict(self) -> dict:
        return {
            "tenant": self.tenant,
            "index_version": self.index_version,
            "cluster_id": self.cluster_id,
            "pq_resident": self.pq_resident,
            "raw_resident": self.raw_resident,
            "pq_bytes": self.pq_bytes,
            "raw_bytes": self.raw_bytes,
            "access_count": self.access_count,
            "last_access_ns": self.last_access_ns,
            "pinned": self.pinned,
            "refcount": self.refcount,
            "version": self.version,
        }


@dataclass(frozen=True)
class DirectorySnapshot:
    """Immutable point-in-time view; contains never falls back to the hash."""

    osd_id: int
    generation: int
    directory_hash: str
    entries: tuple[DirectoryEntry, ...] = ()

    def contains(self, key: ClusterKey | CacheKey, kind: CacheKind | str | None = None) -> bool:
        ident = _ident(key)
        active_kind = kind if kind is not None else (
            key.kind if isinstance(key, CacheKey) else None)
        for entry in self.entries:
            if entry.ident == ident and entry.resident(active_kind):
                return True
        return False

    def entry(self, key: ClusterKey | CacheKey) -> DirectoryEntry | None:
        ident = _ident(key)
        for entry in self.entries:
            if entry.ident == ident:
                return entry
        return None


@dataclass(frozen=True)
class DirectoryDelta:
    """Incremental heartbeat message between two directory generations."""

    osd_id: int
    base_generation: int
    generation: int
    directory_hash: str
    added: tuple[DirectoryEntry, ...] = ()
    updated: tuple[DirectoryEntry, ...] = ()
    removed: tuple[CacheKey, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


@dataclass
class DDRDirectory:
    """Authoritative residency directory for one OSD's DDR partition."""

    osd_id: int
    _entries: dict[tuple[str, str, int], DirectoryEntry] = field(default_factory=dict)
    _generation: int = 0

    def __post_init__(self) -> None:
        _nonneg("osd_id", self.osd_id)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _bump(self) -> None:
        self._generation += 1

    def lookup(self, key: ClusterKey | CacheKey) -> DirectoryEntry | None:
        return self._entries.get(_ident(key))

    def contains(self, key: ClusterKey | CacheKey, kind: CacheKind | str | None = None) -> bool:
        """True only when the concrete entry says the cluster is resident."""
        entry = self.lookup(key)
        active_kind = kind if kind is not None else (
            key.kind if isinstance(key, CacheKey) else None)
        return entry is not None and entry.resident(active_kind)

    def is_cached(self, key: ClusterKey | CacheKey, kind: CacheKind | str = CacheKind.PQ) -> bool:
        return self.contains(key, kind)

    def version_of(self, key: ClusterKey | CacheKey) -> int:
        entry = self.lookup(key)
        return 0 if entry is None else entry.version

    def record_load(self, key: ClusterKey | CacheKey, *, kind: CacheKind | str,
                    payload_bytes: int, now_ns: int = 0) -> DirectoryEntry:
        """Mark one payload kind resident, updating bytes, access and revision."""
        kind_value = coerce_kind(kind)
        _nonneg("payload_bytes", payload_bytes)
        _nonneg("now_ns", now_ns)
        entry = self._entries.get(_ident(key))
        if entry is None:
            entry = DirectoryEntry(
                tenant=key.tenant,
                index_version=key.index_version,
                cluster_id=key.cluster_id,
            )
            self._entries[entry.ident] = entry
        else:
            entry.version += 1
        if kind_value is CacheKind.PQ:
            entry.pq_resident = True
            entry.pq_bytes = payload_bytes
        else:
            entry.raw_resident = True
            entry.raw_bytes = payload_bytes
        entry.access_count += 1
        entry.last_access_ns = now_ns
        self._bump()
        return entry

    def record_remove(self, key: ClusterKey | CacheKey, *, kind: CacheKind | str | None = None,
                      now_ns: int = 0) -> bool:
        """Drop one payload kind, or the whole cluster when both kinds are gone."""
        _nonneg("now_ns", now_ns)
        entry = self._entries.get(_ident(key))
        if entry is None:
            return False
        if kind is None:
            del self._entries[entry.ident]
            self._bump()
            return True
        kind_value = coerce_kind(kind)
        if not entry.resident(kind_value):
            return False
        if kind_value is CacheKind.PQ:
            entry.pq_resident = False
            entry.pq_bytes = 0
        else:
            entry.raw_resident = False
            entry.raw_bytes = 0
        entry.version += 1
        if not entry.resident():
            del self._entries[entry.ident]
        self._bump()
        return True

    def touch(self, key: ClusterKey | CacheKey, now_ns: int = 0,
              access_count: int = 1) -> DirectoryEntry | None:
        entry = self._entries.get(_ident(key))
        if entry is None:
            return None
        _nonneg("now_ns", now_ns)
        _nonneg("access_count", access_count)
        entry.access_count += access_count
        entry.last_access_ns = now_ns
        self._bump()
        return entry

    def pin(self, key: ClusterKey | CacheKey) -> DirectoryEntry | None:
        entry = self._entries.get(_ident(key))
        if entry is None:
            return None
        if not entry.pinned:
            entry.pinned = True
            self._bump()
        return entry

    def unpin(self, key: ClusterKey | CacheKey) -> DirectoryEntry | None:
        entry = self._entries.get(_ident(key))
        if entry is None:
            return None
        if entry.pinned:
            entry.pinned = False
            self._bump()
        return entry

    def acquire(self, key: ClusterKey | CacheKey) -> DirectoryEntry | None:
        entry = self._entries.get(_ident(key))
        if entry is None:
            return None
        entry.refcount += 1
        self._bump()
        return entry

    def release(self, key: ClusterKey | CacheKey) -> DirectoryEntry | None:
        entry = self._entries.get(_ident(key))
        if entry is None or entry.refcount == 0:
            return None
        entry.refcount -= 1
        self._bump()
        return entry

    def evictable(self, key: ClusterKey | CacheKey) -> bool:
        entry = self._entries.get(_ident(key))
        return entry is not None and entry.is_evictable

    def resident(self, *, tenant: str | None = None, index_version: str | None = None,
                 kind: CacheKind | str | None = None) -> tuple[DirectoryEntry, ...]:
        entries = [
            entry
            for entry in self._entries.values()
            if (tenant is None or entry.tenant == tenant)
            and (index_version is None or entry.index_version == index_version)
            and entry.resident(kind)
        ]
        return tuple(sorted(entries, key=lambda entry: entry.ident))

    def snapshot(self) -> DirectorySnapshot:
        """Deep enough copy that later mutations never rewrite the snapshot."""
        return DirectorySnapshot(
            osd_id=self.osd_id,
            generation=self._generation,
            directory_hash=self.directory_hash(),
            entries=tuple(replace(entry) for entry in self.resident()),
        )

    def delta_since(self, base: DirectorySnapshot | DirectoryDelta) -> DirectoryDelta:
        base_entries: dict[tuple[str, str, int], DirectoryEntry] = {}
        if isinstance(base, DirectorySnapshot):
            base_generation = base.generation
            for entry in base.entries:
                base_entries[entry.ident] = entry
        else:
            base_generation = base.generation
            for entry in base.added + base.updated:
                base_entries[entry.ident] = entry
        current = {entry.ident: entry for entry in self.resident()}
        added: list[DirectoryEntry] = []
        updated: list[DirectoryEntry] = []
        for ident, entry in current.items():
            previous = base_entries.get(ident)
            if previous is None:
                added.append(replace(entry))
            elif previous.to_dict() != entry.to_dict():
                updated.append(replace(entry))
        removed = tuple(
            entry.cache_key()
            for ident, entry in base_entries.items()
            if ident not in current
        )
        return DirectoryDelta(
            osd_id=self.osd_id,
            base_generation=base_generation,
            generation=self._generation,
            directory_hash=self.directory_hash(),
            added=tuple(added),
            updated=tuple(updated),
            removed=removed,
        )

    def directory_hash(self) -> str:
        """Canonical sha256 over resident facts; integrity/change detection only."""
        payload = json.dumps(
            [entry.to_dict() for entry in self.resident()],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        self._entries.clear()
        self._bump()

    def telemetry_keys(self, state_of: Callable[[DirectoryEntry | None], str] | None = None):
        """Heartbeat-ready ``ResidentKey`` values, one per resident payload kind."""
        from strata_sim.telemetry.service import KeyKind, ResidentKey, ResidentState

        keys: list[ResidentKey] = []
        for entry in self.resident():
            for kind in CacheKind:
                if not entry.resident(kind):
                    continue
                if state_of is not None:
                    state = ResidentState(state_of(entry))
                elif entry.is_pinned:
                    state = ResidentState.PINNED
                else:
                    state = ResidentState.READY
                keys.append(ResidentKey(
                    key=entry.cluster_id,
                    kind=KeyKind(kind.value),
                    state=state,
                    bytes=entry.bytes_for(kind),
                    tenant=entry.tenant,
                    index_version=entry.index_version,
                    access_count=entry.access_count,
                    last_access_ns=entry.last_access_ns,
                    pinned=entry.pinned,
                    refcount=entry.refcount,
                    entry_version=entry.version,
                ))
        return tuple(keys)
