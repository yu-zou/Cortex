"""Task 13: host-CPU exact rerank over candidate global doc_ids.

Candidates belong to one ``(tenant, index_version, cluster)`` key.  The module
deduplicates them, reads the raw FP32 vectors of *that* key through one of two
paths, and reranks with the frozen float32 L2 oracle and the deterministic
``(distance, doc_id)`` ascending tie-break.

Paths:

* ``SSD``: every needed raw vector is read from SSD once per unique candidate;
* ``DDR``: raw hits come from the board DDR cache, a miss fills the whole
  cluster once (extra SSD/PCIe/DDR bytes are counted) but only the deduplicated
  candidates are reranked;
* ``PQ``: the deduplicated PQ candidate ids are returned without any raw read.

Accounting is explicit: candidate count ``R``, dedup count, merge count, raw
bytes read, cache bytes filled, cache hit/fill counts, DDR bytes read and the
per-phase modeled nanoseconds from an injected :class:`RerankProfile`.  Wall
time is recorded in ``wall_ns`` as a measurement only and is never used as DES
service time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from strata_sim.cache.policy import CacheKey, CacheKind
from strata_sim.cache.store import VersionedCache
from strata_sim.contracts.schema import ClockKind, ProfileRef
from strata_sim.functional.ivfpq import (
    DIM,
    MAX_R,
    RAW_VECTOR_BYTES,
    DatasetRef,
    IdentityMismatchError,
    LayoutError,
    exact_l2_distances,
    pack_raw_vector,
    rank_topk,
    unpack_raw_vector,
)


class RerankPath(str, Enum):
    """Which raw-vector retrieval path a rerank request uses."""

    SSD = "ssd"
    DDR = "ddr"
    PQ = "pq"


class InvalidRawError(LayoutError):
    """A raw vector is truncated, missing bytes, or contains non-finite values."""


class RawNotFoundError(LayoutError):
    """A candidate has no raw vector under the requested cluster key."""


class RerankTimeoutError(ValueError):
    """The modeled elapsed service time exceeds the injected deadline."""


def _require_nonneg(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")


def _require_positive(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


@dataclass(frozen=True)
class RerankProfile:
    """Injected timing profile: bytes/candidates to integer nanoseconds."""

    profile_id: str
    cpu_ns_per_candidate: int
    ssd_command_ns: int
    ssd_bytes_per_ns: int
    pcie_bytes_per_ns: int
    ddr_bytes_per_ns: int
    merge_ns_per_candidate: int = 0
    source: str = "unit-test"
    evidence_id: str = "task-13-parity"

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError(f"profile_id must be a non-empty string, got {self.profile_id!r}")
        _require_nonneg("cpu_ns_per_candidate", self.cpu_ns_per_candidate)
        _require_nonneg("merge_ns_per_candidate", self.merge_ns_per_candidate)
        _require_nonneg("ssd_command_ns", self.ssd_command_ns)
        _require_positive("ssd_bytes_per_ns", self.ssd_bytes_per_ns)
        _require_positive("pcie_bytes_per_ns", self.pcie_bytes_per_ns)
        _require_positive("ddr_bytes_per_ns", self.ddr_bytes_per_ns)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError(f"source must be a non-empty string, got {self.source!r}")
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError(f"evidence_id must be a non-empty string, got {self.evidence_id!r}")

    def to_profile_ref(self) -> ProfileRef:
        """Bind the timing profile to the frozen DES-clock profile contract."""
        return ProfileRef(
            profile_id=self.profile_id,
            clock_kind=ClockKind.DES_LOGICAL,
            source=self.source,
            evidence_id=self.evidence_id,
        )


@dataclass(frozen=True)
class RerankAccounting:
    """Explicit byte and modeled-nanosecond accounting for one rerank."""

    candidate_count: int
    dedup_count: int
    merge_count: int
    raw_bytes_read: int
    cache_bytes_filled: int
    cache_fill_count: int
    cache_hit_count: int
    ddr_bytes_read: int
    ssd_ns: int
    pcie_ns: int
    ddr_ns: int
    cpu_ns: int
    elapsed_ns: int
    wall_ns: int

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_count": self.candidate_count,
            "dedup_count": self.dedup_count,
            "merge_count": self.merge_count,
            "raw_bytes_read": self.raw_bytes_read,
            "cache_bytes_filled": self.cache_bytes_filled,
            "cache_fill_count": self.cache_fill_count,
            "cache_hit_count": self.cache_hit_count,
            "ddr_bytes_read": self.ddr_bytes_read,
            "ssd_ns": self.ssd_ns,
            "pcie_ns": self.pcie_ns,
            "ddr_ns": self.ddr_ns,
            "cpu_ns": self.cpu_ns,
            "elapsed_ns": self.elapsed_ns,
            "wall_ns": self.wall_ns,
        }


@dataclass(frozen=True)
class RerankResult:
    """Exact-reranked top-k plus the path and its accounting."""

    path: RerankPath
    doc_ids: np.ndarray
    distances: np.ndarray
    accounting: RerankAccounting

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.value,
            "doc_ids": [int(doc) for doc in self.doc_ids.tolist()],
            "distances": [float(distance) for distance in self.distances.tolist()],
            "accounting": self.accounting.to_dict(),
        }


class RawVectorStore:
    """Versioned raw-vector source keyed by cluster and global doc_id.

    ``clusters`` maps a cluster id to its member doc_ids in storage order;
    ``vectors`` maps a doc_id to its packed 3072-byte raw vector.  Tests may
    inject truncated or NaN vectors to exercise the fail-closed checks.
    """

    def __init__(
        self,
        ref: DatasetRef,
        clusters: Mapping[int, Sequence[int]],
        vectors: Mapping[int, bytes | bytearray | memoryview],
        cluster: int | None = None,
    ) -> None:
        if not isinstance(ref, DatasetRef):
            raise LayoutError(f"ref must be a DatasetRef, got {ref!r}")
        self.ref = ref
        self.cluster = ref.cluster if cluster is None else int(cluster)
        self.clusters = {
            int(cluster_id): tuple(int(doc) for doc in docs)
            for cluster_id, docs in clusters.items()
        }
        self.vectors = {int(doc): bytes(payload) for doc, payload in vectors.items()}
        if self.cluster is not None and self.cluster not in self.clusters:
            raise RawNotFoundError(f"cluster {self.cluster} has no vectors in this store")

    @classmethod
    def from_arrays(
        cls,
        ref: DatasetRef,
        raw: Any,
        doc_ids: Any,
        assign: Any,
        cluster: int | None = None,
    ) -> "RawVectorStore":
        """Build a store from a ``(n, DIM)`` float32 matrix plus cluster assignment."""
        matrix = np.asarray(raw, dtype=np.float32)
        ids = np.asarray(doc_ids, dtype=np.int64).reshape(-1)
        clusters_assigned = np.asarray(assign, dtype=np.int64).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[1] != DIM:
            raise LayoutError(f"raw must have shape (n, {DIM}), got {matrix.shape}")
        if ids.shape != (matrix.shape[0],) or clusters_assigned.shape != ids.shape:
            raise LayoutError(
                f"doc_ids {ids.shape} / assign {clusters_assigned.shape} incompatible "
                f"with raw {matrix.shape}"
            )
        if np.unique(ids).shape[0] != ids.shape[0]:
            raise LayoutError("doc_ids must be unique")
        if ids.size and int(ids.min()) < 0:
            raise LayoutError("doc_ids must be non-negative")
        if clusters_assigned.size and int(clusters_assigned.min()) < 0:
            raise LayoutError("cluster assignments must be non-negative")
        grouped: dict[int, list[int]] = {}
        vectors: dict[int, bytes] = {}
        for row in range(matrix.shape[0]):
            doc = int(ids[row])
            grouped.setdefault(int(clusters_assigned[row]), []).append(doc)
            vectors[doc] = pack_raw_vector(matrix[row])
        clusters = {cluster_id: tuple(sorted(docs)) for cluster_id, docs in grouped.items()}
        return cls(ref, clusters, vectors, cluster=cluster)

    @classmethod
    def from_index(cls, index: Any, cluster: int | None = None) -> "RawVectorStore":
        """Build a store from an :class:`IVFPQIndex` carrying raw base vectors."""
        if getattr(index, "raw", None) is None:
            raise LayoutError("raw-vector store requires an index with raw base vectors")
        target = index.ref.cluster if cluster is None else cluster
        return cls.from_arrays(index.ref, index.raw, index.doc_ids, index.assign, cluster=target)

    def check_ref(self, ref: DatasetRef) -> None:
        """Fail closed when the request tenant / version / cluster is wrong."""
        if not isinstance(ref, DatasetRef):
            raise LayoutError(f"ref must be a DatasetRef, got {ref!r}")
        if ref.tenant != self.ref.tenant or ref.index_version != self.ref.index_version:
            raise IdentityMismatchError(
                f"requested tenant={ref.tenant!r} index_version={ref.index_version!r} "
                f"but raw store holds tenant={self.ref.tenant!r} "
                f"index_version={self.ref.index_version!r}"
            )
        if ref.cluster is not None and self.cluster is not None:
            if int(ref.cluster) != self.cluster:
                raise IdentityMismatchError(
                    f"requested cluster={ref.cluster} but raw store holds cluster={self.cluster}"
                )

    def resolve_cluster(self, ref: DatasetRef | None) -> int:
        if ref is not None and ref.cluster is not None:
            return int(ref.cluster)
        if self.cluster is not None:
            return self.cluster
        if len(self.clusters) == 1:
            return next(iter(self.clusters))
        raise LayoutError("cluster must be specified when the raw store spans multiple clusters")

    def cluster_doc_ids(self, cluster: int) -> tuple[int, ...]:
        members = self.clusters.get(int(cluster))
        if members is None:
            raise RawNotFoundError(
                f"cluster {cluster} is not present for tenant={self.ref.tenant!r} "
                f"index_version={self.ref.index_version!r}"
            )
        return members

    def cluster_size(self, cluster: int) -> int:
        return len(self.cluster_doc_ids(cluster))

    def raw_offset(self, cluster: int, doc_id: int) -> int:
        members = self.cluster_doc_ids(cluster)
        try:
            return members.index(int(doc_id)) * RAW_VECTOR_BYTES
        except ValueError:
            raise RawNotFoundError(
                f"doc_id {doc_id} is not a member of cluster {cluster}"
            ) from None

    def raw_vector_bytes(self, cluster: int, doc_id: int) -> bytes:
        self.raw_offset(cluster, doc_id)
        payload = self.vectors.get(int(doc_id))
        if payload is None:
            raise RawNotFoundError(f"doc_id {doc_id} has no stored raw vector")
        return payload

    def cluster_blob(self, cluster: int) -> bytes:
        """Whole-cluster raw bytes in storage order, length-checked per member."""
        members = self.cluster_doc_ids(cluster)
        parts: list[bytes] = []
        for doc in members:
            payload = self.vectors.get(doc)
            if payload is None:
                raise RawNotFoundError(f"doc_id {doc} has no stored raw vector")
            if len(payload) != RAW_VECTOR_BYTES:
                raise InvalidRawError(
                    f"raw vector for doc_id {doc} is {len(payload)} bytes, "
                    f"expected {RAW_VECTOR_BYTES}"
                )
            parts.append(payload)
        return b"".join(parts)


def _require_profile(profile: RerankProfile) -> None:
    if not isinstance(profile, RerankProfile):
        raise ValueError(f"profile must be a RerankProfile, got {type(profile).__name__}")


def _transfer_ns(nbytes: int, bytes_per_ns: int) -> int:
    if nbytes <= 0:
        return 0
    return -(-nbytes // bytes_per_ns)


def _unpack_checked(payload: bytes, doc_id: int) -> np.ndarray:
    if len(payload) != RAW_VECTOR_BYTES:
        raise InvalidRawError(
            f"raw vector for doc_id {doc_id} is {len(payload)} bytes, "
            f"expected {RAW_VECTOR_BYTES}"
        )
    vector = unpack_raw_vector(payload)
    if not bool(np.isfinite(vector).all()):
        raise InvalidRawError(f"raw vector for doc_id {doc_id} contains non-finite values")
    return vector


def _dedup_candidates(candidates: Any) -> tuple[np.ndarray, int, int, int]:
    ids = np.asarray(candidates, dtype=np.int64).reshape(-1)
    if ids.size and int(ids.min()) < 0:
        raise ValueError(f"doc_ids must be non-negative, got {int(ids.min())}")
    unique = np.unique(ids)
    candidate_count = int(ids.size)
    dedup_count = int(unique.size)
    if dedup_count > MAX_R:
        raise ValueError(
            f"candidate list holds {dedup_count} unique ids > MAX_R={MAX_R}; "
            "refusing to silently truncate"
        )
    return unique, candidate_count, dedup_count, candidate_count - dedup_count


def _stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, DIM), dtype=np.float32)
    return np.stack(rows).astype(np.float32, copy=False)


def _finalize(
    path: RerankPath,
    unique: np.ndarray,
    rows: Sequence[np.ndarray],
    query: Any,
    k: int,
    *,
    candidate_count: int,
    dedup_count: int,
    merge_count: int,
    raw_bytes_read: int,
    cache_bytes_filled: int,
    cache_fill_count: int,
    cache_hit_count: int,
    ddr_bytes_read: int,
    ssd_ns: int,
    pcie_ns: int,
    ddr_ns: int,
    cpu_ns: int,
    deadline_ns: int | None,
    record_wall: bool,
    started_ns: int,
) -> RerankResult:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")
    matrix = _stack(rows)
    if unique.size:
        distances = exact_l2_distances(query, matrix)
        doc_ids, top_distances = rank_topk(distances, unique, k)
    else:
        doc_ids = np.empty(0, dtype=np.int64)
        top_distances = np.empty(0, dtype=np.float32)
    elapsed_ns = ssd_ns + pcie_ns + ddr_ns + cpu_ns
    if deadline_ns is not None:
        _require_nonneg("deadline_ns", deadline_ns)
        if elapsed_ns > deadline_ns:
            raise RerankTimeoutError(
                f"modeled rerank elapsed {elapsed_ns}ns exceeds deadline {deadline_ns}ns"
            )
    accounting = RerankAccounting(
        candidate_count=candidate_count,
        dedup_count=dedup_count,
        merge_count=merge_count,
        raw_bytes_read=raw_bytes_read,
        cache_bytes_filled=cache_bytes_filled,
        cache_fill_count=cache_fill_count,
        cache_hit_count=cache_hit_count,
        ddr_bytes_read=ddr_bytes_read,
        ssd_ns=ssd_ns,
        pcie_ns=pcie_ns,
        ddr_ns=ddr_ns,
        cpu_ns=cpu_ns,
        elapsed_ns=elapsed_ns,
        wall_ns=(time.perf_counter_ns() - started_ns) if record_wall else 0,
    )
    return RerankResult(
        path=path, doc_ids=doc_ids, distances=top_distances, accounting=accounting
    )


def rerank_from_ssd(
    candidates: Any,
    query: Any,
    store: RawVectorStore,
    *,
    ref: DatasetRef,
    k: int,
    profile: RerankProfile,
    deadline_ns: int | None = None,
    record_wall: bool = True,
) -> RerankResult:
    """SSD-only exact rerank: read each unique candidate vector once."""
    _require_profile(profile)
    store.check_ref(ref)
    cluster = store.resolve_cluster(ref)
    unique, candidate_count, dedup_count, merge_count = _dedup_candidates(candidates)
    started_ns = time.perf_counter_ns()
    rows: list[np.ndarray] = []
    raw_bytes_read = 0
    for doc in unique.tolist():
        payload = store.raw_vector_bytes(cluster, doc)
        raw_bytes_read += len(payload)
        rows.append(_unpack_checked(payload, doc))
    ssd_ns = profile.ssd_command_ns + _transfer_ns(raw_bytes_read, profile.ssd_bytes_per_ns)
    cpu_ns = (
        profile.cpu_ns_per_candidate * dedup_count
        + profile.merge_ns_per_candidate * candidate_count
    )
    return _finalize(
        RerankPath.SSD,
        unique,
        rows,
        query,
        k,
        candidate_count=candidate_count,
        dedup_count=dedup_count,
        merge_count=merge_count,
        raw_bytes_read=raw_bytes_read,
        cache_bytes_filled=0,
        cache_fill_count=0,
        cache_hit_count=0,
        ddr_bytes_read=0,
        ssd_ns=ssd_ns,
        pcie_ns=0,
        ddr_ns=0,
        cpu_ns=cpu_ns,
        deadline_ns=deadline_ns,
        record_wall=record_wall,
        started_ns=started_ns,
    )


def rerank_from_ddr_cache(
    candidates: Any,
    query: Any,
    store: RawVectorStore,
    *,
    ref: DatasetRef,
    k: int,
    profile: RerankProfile,
    cache: VersionedCache,
    deadline_ns: int | None = None,
    record_wall: bool = True,
) -> RerankResult:
    """DDR-cache exact rerank: hits come from cache, a miss fills the whole cluster."""
    _require_profile(profile)
    if not isinstance(cache, VersionedCache):
        raise ValueError(f"cache must be a VersionedCache, got {type(cache).__name__}")
    store.check_ref(ref)
    cluster = store.resolve_cluster(ref)
    unique, candidate_count, dedup_count, merge_count = _dedup_candidates(candidates)
    started_ns = time.perf_counter_ns()
    rows: list[np.ndarray] = []
    raw_bytes_read = 0
    cache_bytes_filled = 0
    cache_fill_count = 0
    cache_hit_count = 0
    if unique.size:
        key = CacheKey(store.ref.tenant, store.ref.index_version, cluster, CacheKind.RAW)
        blob = cache.get(key)
        if blob is None:
            blob = cache.load(key, lambda _key: store.cluster_blob(cluster))
            cache_bytes_filled = len(blob)
            raw_bytes_read = cache_bytes_filled
            cache_fill_count = 1
        else:
            cache_hit_count = dedup_count
        expected = store.cluster_size(cluster) * RAW_VECTOR_BYTES
        if len(blob) != expected:
            raise InvalidRawError(
                f"cached raw blob for cluster {cluster} is {len(blob)} bytes, "
                f"expected {expected}"
            )
        for doc in unique.tolist():
            offset = store.raw_offset(cluster, doc)
            rows.append(_unpack_checked(blob[offset : offset + RAW_VECTOR_BYTES], doc))
    ddr_bytes_read = dedup_count * RAW_VECTOR_BYTES
    if cache_bytes_filled:
        ssd_ns = profile.ssd_command_ns + _transfer_ns(
            cache_bytes_filled, profile.ssd_bytes_per_ns
        )
    else:
        ssd_ns = 0
    pcie_ns = _transfer_ns(cache_bytes_filled, profile.pcie_bytes_per_ns)
    ddr_ns = _transfer_ns(cache_bytes_filled, profile.ddr_bytes_per_ns) + _transfer_ns(
        ddr_bytes_read, profile.ddr_bytes_per_ns
    )
    cpu_ns = (
        profile.cpu_ns_per_candidate * dedup_count
        + profile.merge_ns_per_candidate * candidate_count
    )
    return _finalize(
        RerankPath.DDR,
        unique,
        rows,
        query,
        k,
        candidate_count=candidate_count,
        dedup_count=dedup_count,
        merge_count=merge_count,
        raw_bytes_read=raw_bytes_read,
        cache_bytes_filled=cache_bytes_filled,
        cache_fill_count=cache_fill_count,
        cache_hit_count=cache_hit_count,
        ddr_bytes_read=ddr_bytes_read,
        ssd_ns=ssd_ns,
        pcie_ns=pcie_ns,
        ddr_ns=ddr_ns,
        cpu_ns=cpu_ns,
        deadline_ns=deadline_ns,
        record_wall=record_wall,
        started_ns=started_ns,
    )


def rerank_pq_only(
    candidates: Any,
    *,
    profile: RerankProfile,
    record_wall: bool = True,
) -> RerankResult:
    """PQ-only mode: deduplicated candidate ids with zero raw reads."""
    _require_profile(profile)
    started_ns = time.perf_counter_ns()
    unique, candidate_count, dedup_count, merge_count = _dedup_candidates(candidates)
    merge_ns = profile.merge_ns_per_candidate * candidate_count
    accounting = RerankAccounting(
        candidate_count=candidate_count,
        dedup_count=dedup_count,
        merge_count=merge_count,
        raw_bytes_read=0,
        cache_bytes_filled=0,
        cache_fill_count=0,
        cache_hit_count=0,
        ddr_bytes_read=0,
        ssd_ns=0,
        pcie_ns=0,
        ddr_ns=0,
        cpu_ns=merge_ns,
        elapsed_ns=merge_ns,
        wall_ns=(time.perf_counter_ns() - started_ns) if record_wall else 0,
    )
    return RerankResult(
        path=RerankPath.PQ,
        doc_ids=unique,
        distances=np.empty(0, dtype=np.float32),
        accounting=accounting,
    )


def rerank(
    candidates: Any,
    query: Any,
    store: RawVectorStore | None,
    *,
    ref: DatasetRef | None,
    k: int,
    path: RerankPath | str,
    profile: RerankProfile,
    cache: VersionedCache | None = None,
    deadline_ns: int | None = None,
    record_wall: bool = True,
) -> RerankResult:
    """Dispatch to the SSD, DDR or PQ path with one shared accounting shape."""
    resolved = path if isinstance(path, RerankPath) else RerankPath(path)
    if resolved is RerankPath.SSD:
        if store is None or ref is None:
            raise ValueError("the SSD path requires a raw store and a DatasetRef")
        return rerank_from_ssd(
            candidates,
            query,
            store,
            ref=ref,
            k=k,
            profile=profile,
            deadline_ns=deadline_ns,
            record_wall=record_wall,
        )
    if resolved is RerankPath.DDR:
        if store is None or ref is None or cache is None:
            raise ValueError("the DDR path requires a raw store, a DatasetRef and a cache")
        return rerank_from_ddr_cache(
            candidates,
            query,
            store,
            ref=ref,
            k=k,
            profile=profile,
            cache=cache,
            deadline_ns=deadline_ns,
            record_wall=record_wall,
        )
    return rerank_pq_only(candidates, profile=profile, record_wall=record_wall)


__all__ = [
    "InvalidRawError",
    "RawNotFoundError",
    "RawVectorStore",
    "RerankAccounting",
    "RerankPath",
    "RerankProfile",
    "RerankResult",
    "RerankTimeoutError",
    "rerank",
    "rerank_from_ddr_cache",
    "rerank_from_ssd",
    "rerank_pq_only",
]
