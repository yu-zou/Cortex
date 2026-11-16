"""Task 2: frozen data, clock and experiment contracts.

Rules encoded here (fail closed):
- units are explicit: ns for time, bytes for sizes, cycles only in HLS reports;
- ``des_logical`` events can never carry a wall timestamp and vice versa;
- profiles must cite a source and an evidence id and must use the DES clock;
- a node may only serve a cluster whose index version is actually READY/PINNED;
- unknown schema versions are rejected, never silently migrated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1
TILE_MAX_DOCS = 8192
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A contract invariant was violated."""


class UnsupportedSchemaError(ContractError):
    """The document schema version is not this code's version."""


class ClockKind(str, Enum):
    DES_LOGICAL = "des_logical"
    WALL = "wall"


class EventPhase(str, Enum):
    ARRIVAL = "arrival"
    ADMIT = "admit"
    SCHEDULE = "schedule"
    QUEUE = "queue"
    LOAD = "load"
    COMPUTE = "compute"
    RERANK = "rerank"
    MERGE = "merge"
    COMPLETE = "complete"


class VersionState(str, Enum):
    ABSENT = "absent"
    LOADING = "loading"
    READY = "ready"
    PINNED = "pinned"
    EVICTED = "evicted"


_SERVABLE = {VersionState.READY, VersionState.PINNED}


def _check_nonneg(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative int, got {value!r}")


def _check_sha256(name: str, value: str | None) -> None:
    if value is not None and not _HEX64.match(value):
        raise ContractError(f"{name} must be a 64-char lowercase hex sha256, got {value!r}")


def _check_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string, got {value!r}")


@dataclass(frozen=True)
class ClusterKey:
    tenant: str
    index_version: str
    cluster_id: int

    def __post_init__(self) -> None:
        _check_str("tenant", self.tenant)
        _check_str("index_version", self.index_version)
        if not isinstance(self.cluster_id, int) or self.cluster_id < 0:
            raise ContractError(f"cluster_id must be a non-negative int, got {self.cluster_id!r}")

    def to_dict(self) -> dict:
        return {"tenant": self.tenant, "index_version": self.index_version,
                "cluster_id": self.cluster_id}

    @classmethod
    def from_dict(cls, d: dict) -> "ClusterKey":
        return cls(tenant=d["tenant"], index_version=d["index_version"],
                   cluster_id=d["cluster_id"])


@dataclass(frozen=True)
class DesEvent:
    time_ns: int
    phase: EventPhase
    sequence: int
    clock_kind: ClockKind = ClockKind.DES_LOGICAL

    def __post_init__(self) -> None:
        _check_nonneg("time_ns", self.time_ns)
        _check_nonneg("sequence", self.sequence)
        if self.clock_kind is ClockKind.WALL or str(self.clock_kind) == ClockKind.WALL.value:
            raise ContractError("DES events must use clock_kind=des_logical, not wall")
        if not isinstance(self.phase, EventPhase):
            raise ContractError(f"phase must be an EventPhase, got {self.phase!r}")

    def to_dict(self) -> dict:
        return {"time_ns": self.time_ns, "phase": self.phase.value,
                "sequence": self.sequence, "clock_kind": ClockKind.DES_LOGICAL.value}

    @classmethod
    def from_dict(cls, d: dict) -> "DesEvent":
        kind = d.get("clock_kind", ClockKind.DES_LOGICAL.value)
        if kind != ClockKind.DES_LOGICAL.value:
            raise ContractError(f"refusing non-DES clock_kind {kind!r}")
        return cls(time_ns=d["time_ns"], phase=EventPhase(d["phase"]),
                   sequence=d["sequence"], clock_kind=ClockKind.DES_LOGICAL)


def event_sort_key(event: DesEvent) -> tuple[int, int, int]:
    """Total, deterministic ordering: (time, phase order, sequence)."""
    return (event.time_ns, list(EventPhase).index(event.phase), event.sequence)


@dataclass(frozen=True)
class ObjectRef:
    object_id: str
    offset: int
    length: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        _check_str("object_id", self.object_id)
        _check_nonneg("offset", self.offset)
        _check_nonneg("length", self.length)
        _check_sha256("sha256", self.sha256)

    def to_dict(self) -> dict:
        d = {"object_id": self.object_id, "offset": self.offset, "length": self.length}
        if self.sha256 is not None:
            d["sha256"] = self.sha256
        return d


@dataclass(frozen=True)
class TileRef:
    cluster_key: ClusterKey
    tile_index: int
    global_doc_start: int
    global_doc_count: int

    def __post_init__(self) -> None:
        _check_nonneg("tile_index", self.tile_index)
        _check_nonneg("global_doc_start", self.global_doc_start)
        _check_nonneg("global_doc_count", self.global_doc_count)
        if self.global_doc_count > TILE_MAX_DOCS:
            raise ContractError(
                f"tile holds {self.global_doc_count} docs > {TILE_MAX_DOCS}; split first")


@dataclass(frozen=True)
class ObjectManifest:
    schema_version: int
    dataset_id: str
    metric: str
    dim: int
    dtype: str
    base_count: int
    object_refs: list[ObjectRef] = field(default_factory=list)
    query_count: int = 0
    gt_count: int = 0
    tenant_semantics: str = "constructed-experiment"

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise UnsupportedSchemaError(
                f"manifest schema {self.schema_version} != {SCHEMA_VERSION}")
        _check_str("dataset_id", self.dataset_id)
        _check_str("metric", self.metric)
        _check_str("dtype", self.dtype)
        if self.dim <= 0:
            raise ContractError(f"dim must be positive, got {self.dim}")
        if self.base_count <= 0:
            raise ContractError(f"base_count must be positive, got {self.base_count}")
        _check_nonneg("query_count", self.query_count)
        _check_nonneg("gt_count", self.gt_count)
        if not self.object_refs:
            raise ContractError("manifest must define object_refs")


@dataclass(frozen=True)
class ProfileRef:
    profile_id: str
    clock_kind: ClockKind
    source: str
    evidence_id: str

    def __post_init__(self) -> None:
        _check_str("profile_id", self.profile_id)
        _check_str("source", self.source)
        _check_str("evidence_id", self.evidence_id)
        if self.clock_kind is not ClockKind.DES_LOGICAL:
            raise ContractError(
                "profile timing must be keyed by a DES clock reference, not wall clock")


@dataclass
class NodeInfo:
    osd_id: int
    logical_host: str
    compute_agent: str
    fpga_model: str
    resident_versions: dict[str, VersionState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_nonneg("osd_id", self.osd_id)
        _check_str("logical_host", self.logical_host)
        _check_str("compute_agent", self.compute_agent)
        _check_str("fpga_model", self.fpga_model)
        converted = {}
        for version, state in self.resident_versions.items():
            converted[version] = state if isinstance(state, VersionState) else VersionState(state)
        self.resident_versions = converted

    def can_serve(self, key: ClusterKey) -> bool:
        return self.resident_versions.get(key.index_version) in _SERVABLE


_KIND_LOADERS: dict[str, Any] = {
    "cluster_key": ClusterKey.from_dict,
    "des_event": DesEvent.from_dict,
}


def load_document(doc: dict) -> Any:
    """Validate the envelope and dispatch, failing closed on unknown versions."""
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"unsupported schema_version {version!r}; this build speaks {SCHEMA_VERSION}")
    kind = doc.get("kind")
    if kind not in _KIND_LOADERS:
        raise UnsupportedSchemaError(f"unknown document kind {kind!r}")
    obj = _KIND_LOADERS[kind](doc["payload"])
    return obj.to_dict() if hasattr(obj, "to_dict") else obj


def wrap_document(kind: str, payload: Any) -> dict:
    """Wrap a payload with the current schema envelope (inverse of load_document)."""
    if kind not in _KIND_LOADERS:
        raise UnsupportedSchemaError(f"unknown document kind {kind!r}")
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "payload": payload}
