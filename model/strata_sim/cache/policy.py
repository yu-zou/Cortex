"""Task 10 cache policy: partition budgets, byte-LRU/LFU and hot-cluster prefetch.

Rules:
- PQ and raw payloads get explicit byte budgets derived from configured ratios;
- eviction is byte-LRU (baseline) or byte-LFU over unpinned entries only;
- the prefetch planner may only read trace entries at or before its cursor, so
  it can never consume future trace information;
- ARC is intentionally skipped in this iteration: the task-10 acceptance needs
  byte-LRU plus the LFU option and the extra ghost-list state would not be
  exercised by any acceptance or QA scenario, so it stays documented instead of
  half-implemented.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

from strata_sim.contracts.schema import ContractError

_RATIO_EPSILON = 1e-6


class CacheKind(str, Enum):
    """The two payload classes stored by the cache."""

    PQ = "pq"
    RAW = "raw"


def coerce_kind(kind: CacheKind | str) -> CacheKind:
    """Return a CacheKind, rejecting anything that is not pq/raw."""
    if isinstance(kind, CacheKind):
        return kind
    try:
        return CacheKind(kind)
    except ValueError as exc:
        raise ContractError(f"unknown cache kind {kind!r}") from exc


@dataclass(frozen=True)
class CacheKey:
    """Versioned identity of an entry: (tenant, index_version, cluster_id, kind)."""

    tenant: str
    index_version: str
    cluster_id: int
    kind: CacheKind = CacheKind.PQ

    def __post_init__(self) -> None:
        if not isinstance(self.tenant, str) or not self.tenant:
            raise ContractError(f"tenant must be a non-empty string, got {self.tenant!r}")
        if not isinstance(self.index_version, str) or not self.index_version:
            raise ContractError(
                f"index_version must be a non-empty string, got {self.index_version!r}"
            )
        if not isinstance(self.cluster_id, int) or self.cluster_id < 0:
            raise ContractError(f"cluster_id must be a non-negative int, got {self.cluster_id!r}")
        object.__setattr__(self, "kind", coerce_kind(self.kind))

    def to_dict(self) -> dict:
        return {
            "tenant": self.tenant,
            "index_version": self.index_version,
            "cluster_id": self.cluster_id,
            "kind": self.kind.value,
        }


def normalize_ratios(ratios: Mapping[CacheKind | str, float]) -> dict[CacheKind, float]:
    """Normalize partition ratios given as fractions or percentages to fractions summing to 1."""
    if not ratios:
        raise ContractError("at least one cache partition ratio is required")
    parsed = {coerce_kind(kind): float(value) for kind, value in ratios.items()}
    if any(value < 0.0 for value in parsed.values()):
        raise ContractError(f"cache partition ratios must be >= 0, got {parsed!r}")
    total = sum(parsed.values())
    if total <= _RATIO_EPSILON:
        raise ContractError("cache partition ratios must sum to a positive value")
    scale = 100.0 if total > 1.0 + _RATIO_EPSILON else 1.0
    fractions = {kind: value / scale for kind, value in parsed.items()}
    if abs(sum(fractions.values()) - 1.0) > _RATIO_EPSILON:
        raise ContractError(f"cache partition ratios must sum to 1 or 100, got {ratios!r}")
    return fractions


@dataclass(frozen=True)
class PartitionPlan:
    """Byte budget per cache kind; floor leftovers are left unallocated."""

    capacity_bytes: int
    ratios: Mapping[CacheKind, float]
    budgets: Mapping[CacheKind, int]

    @classmethod
    def from_ratios(
        cls, capacity_bytes: int, ratios: Mapping[CacheKind | str, float]
    ) -> "PartitionPlan":
        if not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
            raise ContractError(f"capacity_bytes must be a positive int, got {capacity_bytes!r}")
        fractions = normalize_ratios(ratios)
        budgets = {kind: int(capacity_bytes * fraction) for kind, fraction in fractions.items()}
        return cls(capacity_bytes=capacity_bytes, ratios=fractions, budgets=budgets)

    def budget_for(self, kind: CacheKind | str) -> int:
        """Bytes available to one kind; kinds absent from the ratios get zero."""
        return self.budgets.get(coerce_kind(kind), 0)


class ByteLRU:
    """LRU ordering with byte accounting for a single budget pool."""

    def __init__(self, budget_bytes: int) -> None:
        if not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ContractError(f"budget_bytes must be a non-negative int, got {budget_bytes!r}")
        self.budget_bytes = budget_bytes
        self._sizes: OrderedDict[CacheKey, int] = OrderedDict()

    @property
    def total_bytes(self) -> int:
        return sum(self._sizes.values())

    def add(self, key: CacheKey, size: int) -> None:
        """Insert a fresh entry and make it the most recently used."""
        self._sizes[key] = size
        self._sizes.move_to_end(key)

    def touch(self, key: CacheKey) -> None:
        """Mark an existing entry as the most recently used."""
        if key in self._sizes:
            self._sizes.move_to_end(key)

    def remove(self, key: CacheKey) -> None:
        self._sizes.pop(key, None)

    def victims_for(
        self, needed_bytes: int, evictable: Callable[[CacheKey], bool]
    ) -> list[CacheKey] | None:
        """Return oldest-first victims freeing room for ``needed_bytes``.

        ``None`` means no set of evictable entries can make room, so the caller
        must reject the insert instead of over-allocating.
        """
        need_to_free = self.total_bytes + needed_bytes - self.budget_bytes
        if need_to_free <= 0:
            return []
        victims: list[CacheKey] = []
        freed = 0
        for key, size in self._sizes.items():
            if freed >= need_to_free:
                break
            if evictable(key):
                victims.append(key)
                freed += size
        if freed < need_to_free:
            return None
        return victims


class ByteLFU:
    """Least-frequently-used ordering with byte accounting for one budget pool.

    Frequency ties break towards the least recently touched entry so a crafted
    hot/cold pattern is deterministic.
    """

    def __init__(self, budget_bytes: int) -> None:
        if not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ContractError(f"budget_bytes must be a non-negative int, got {budget_bytes!r}")
        self.budget_bytes = budget_bytes
        self._sizes: dict[CacheKey, int] = {}
        self._frequency: dict[CacheKey, int] = {}
        self._recency: dict[CacheKey, int] = {}
        self._clock = 0

    @property
    def total_bytes(self) -> int:
        return sum(self._sizes.values())

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def add(self, key: CacheKey, size: int) -> None:
        """Insert a fresh entry with frequency one."""
        self._sizes[key] = size
        self._frequency[key] = 1
        self._recency[key] = self._tick()

    def touch(self, key: CacheKey) -> None:
        """Record one access; frequency drives eviction, recency breaks ties."""
        if key in self._sizes:
            self._frequency[key] += 1
            self._recency[key] = self._tick()

    def remove(self, key: CacheKey) -> None:
        self._sizes.pop(key, None)
        self._frequency.pop(key, None)
        self._recency.pop(key, None)

    def victims_for(
        self, needed_bytes: int, evictable: Callable[[CacheKey], bool]
    ) -> list[CacheKey] | None:
        """Return least-frequent-first victims freeing room for ``needed_bytes``."""
        need_to_free = self.total_bytes + needed_bytes - self.budget_bytes
        if need_to_free <= 0:
            return []
        ranked = sorted(
            self._sizes,
            key=lambda key: (self._frequency.get(key, 0), self._recency.get(key, 0)),
        )
        victims: list[CacheKey] = []
        freed = 0
        for key in ranked:
            if freed >= need_to_free:
                break
            if evictable(key):
                victims.append(key)
                freed += self._sizes[key]
        if freed < need_to_free:
            return None
        return victims


@dataclass(frozen=True)
class PrefetchPlan:
    """Clusters selected for prefetch from observed traffic only."""

    cluster_ids: tuple[int, ...]

    def __bool__(self) -> bool:
        return bool(self.cluster_ids)

    def __iter__(self):
        return iter(self.cluster_ids)


class HotClusterPrefetcher:
    """Ranks clusters in the already-observed trace prefix by hit count."""

    def __init__(self, lookback: int = 4, min_hits: int = 2, max_candidates: int = 4) -> None:
        if not isinstance(lookback, int) or lookback < 1:
            raise ContractError(f"lookback must be a positive int, got {lookback!r}")
        if not isinstance(min_hits, int) or min_hits < 1:
            raise ContractError(f"min_hits must be a positive int, got {min_hits!r}")
        if not isinstance(max_candidates, int) or max_candidates < 1:
            raise ContractError(f"max_candidates must be a positive int, got {max_candidates!r}")
        self.lookback = lookback
        self.min_hits = min_hits
        self.max_candidates = max_candidates

    def candidates(
        self,
        trace: Sequence[int],
        cursor: int,
        resident_clusters: Iterable[int] = (),
    ) -> tuple[int, ...]:
        """Hot clusters from ``trace[:cursor + 1]`` only, excluding the current one."""
        if not isinstance(cursor, int) or cursor < 0 or cursor >= len(trace):
            raise ContractError(f"cursor {cursor!r} outside trace of length {len(trace)}")
        start = max(0, cursor - self.lookback + 1)
        window = [trace[index] for index in range(start, cursor + 1)]
        counts: dict[int, int] = {}
        for cluster_id in window:
            counts[cluster_id] = counts.get(cluster_id, 0) + 1
        current = trace[cursor]
        resident = set(resident_clusters)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        selected = [
            cluster_id
            for cluster_id, hits in ranked
            if hits >= self.min_hits and cluster_id != current and cluster_id not in resident
        ]
        return tuple(selected[: self.max_candidates])
