"""Functional (CPU) oracle for the Strata IVFPQ residual-L2 path.

Semantics frozen here (see ``docs/data.md`` for the audit that motivated them):

* Coarse quantizer: L2, ``nlist`` centroids of ``DIM=768`` FP32 each.
* Query residual against a probed cluster ``c``: ``q - centroid[c]``.
* PQ ADC: for each subquantizer ``m`` build a distance table row over the
  ``KS=256`` codebook entries of the residual query part and sum over ``m``.
* Exact ground truth: full-precision float32 L2 over the raw base vectors
  (never the PQ codes), top-k with deterministic ``(distance, doc_id)``
  ascending tie-break.
* Tiling: exact scans split into tiles of at most ``TILE_SIZE=8192`` docs and
  merge tile candidates by global ``doc_id``; per-cluster candidate lists are
  capped at ``r`` with ``r <= MAX_R=500`` enforced (never silently truncated).

Byte layouts:

* raw vector: ``768 * 4 = 3072`` bytes little-endian float32.
* PQ hardware entry: ``128`` bytes = ``96`` code bytes + ``16`` metadata bytes
  + ``16`` zero pad bytes. Metadata = uint64 LE doc_id, uint32 LE doc_len,
  uint32 LE flags. The 1B R5 artifact predates this contract (8-byte doc_id +
  24 pad bytes); ``read_cluster_header`` documents the observed layout.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from strata_sim.safety import DEFAULT_PROJECT_ROOT

DIM = 768
M = 96
NBITS = 8
KS = 1 << NBITS
DS = DIM // M
RAW_VECTOR_BYTES = DIM * 4
CODE_BYTES = M
META_BYTES = 16
PAD_BYTES = 16
ENTRY_BYTES = CODE_BYTES + META_BYTES + PAD_BYTES
CLUSTER_HEADER_BYTES = 64
TILE_SIZE = 8192
MAX_R = 500
DEFAULT_K = 10
DEFAULT_R = 64
DEFAULT_NPROBE = 8

_HASH_CHUNK = 1 << 20


class IdentityMismatchError(ValueError):
    """Raised when a tenant / index_version / cluster key does not match."""


class LayoutError(ValueError):
    """Raised when an array or byte buffer violates the frozen layout."""


@dataclass(frozen=True)
class DatasetRef:
    """Identity of the dataset a query is allowed to touch."""

    tenant: str
    index_version: str
    cluster: int | None = None


@dataclass
class SearchResult:
    """Exact-reranked top-k plus the merged ADC candidate list."""

    doc_ids: np.ndarray
    distances: np.ndarray
    candidate_doc_ids: np.ndarray
    candidate_distances: np.ndarray
    candidate_rows: np.ndarray


def residual_query(query: Any, centroid: Any) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    c = np.asarray(centroid, dtype=np.float32).reshape(-1)
    if q.shape[0] != c.shape[0]:
        raise LayoutError(f"query dim {q.shape[0]} != centroid dim {c.shape[0]}")
    return (q - c).astype(np.float32)


def pq_distance_table(residual: Any, codebook: Any) -> np.ndarray:
    """Return the (M, KS) float32 table of residual-part L2 distances."""
    res = np.asarray(residual, dtype=np.float32).reshape(-1)
    cb = np.asarray(codebook, dtype=np.float32)
    if cb.ndim != 3:
        raise LayoutError(f"codebook must be 3-D, got shape {cb.shape}")
    m, ks, ds = cb.shape
    if res.shape[0] != m * ds:
        raise LayoutError(f"residual size {res.shape[0]} != M*DS {m * ds}")
    qr = res.reshape(m, ds)
    table = np.empty((m, ks), dtype=np.float32)
    for mi in range(m):
        diff = cb[mi] - qr[mi]
        table[mi] = np.sum(diff * diff, axis=1, dtype=np.float32)
    return table


def pq_adc_distances(table: Any, codes: Any) -> np.ndarray:
    """Sum table[m, codes[:, m]] over m -> (n,) float32 distances."""
    tbl = np.asarray(table, dtype=np.float32)
    if tbl.ndim != 2:
        raise LayoutError(f"distance table must be 2-D, got shape {tbl.shape}")
    cds = np.asarray(codes, dtype=np.uint8)
    if cds.ndim != 2 or cds.shape[1] != tbl.shape[0]:
        raise LayoutError(f"codes shape {cds.shape} incompatible with table {tbl.shape}")
    out = np.zeros(cds.shape[0], dtype=np.float32)
    for mi in range(tbl.shape[0]):
        out += tbl[mi][cds[:, mi]]
    return out


def exact_l2_distances(query: Any, raw: Any) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    r = np.asarray(raw, dtype=np.float32)
    if r.ndim != 2 or r.shape[1] != q.shape[0]:
        raise LayoutError(f"raw shape {r.shape} incompatible with query dim {q.shape[0]}")
    diff = r - q
    return np.sum(diff * diff, axis=1, dtype=np.float32)


def rank_topk(distances: Any, doc_ids: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic top-k by ``(distance, doc_id)`` ascending."""
    dists = np.asarray(distances)
    ids = np.asarray(doc_ids)
    if dists.shape[0] != ids.shape[0]:
        raise LayoutError(f"distances {dists.shape} != doc_ids {ids.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if dists.shape[0] == 0:
        return ids.astype(np.int64), dists
    if dists.ndim != 1 or ids.ndim != 1:
        raise LayoutError("rank_topk expects 1-D distances and doc_ids")
    take = min(k, dists.shape[0])
    order = np.lexsort((ids, dists))[:take]
    return ids[order], dists[order]


def scan_tiled(
    raw: Any,
    doc_ids: Any,
    query: Any,
    k: int,
    tile_size: int = TILE_SIZE,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Exact float32 L2 top-k over tiles of at most ``tile_size`` docs."""
    r = np.asarray(raw, dtype=np.float32)
    ids = np.asarray(doc_ids, dtype=np.int64)
    if r.ndim != 2:
        raise LayoutError(f"raw must be 2-D, got shape {r.shape}")
    if ids.shape != (r.shape[0],):
        raise LayoutError(f"doc_ids shape {ids.shape} != ({r.shape[0]},)")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (1 <= tile_size <= TILE_SIZE):
        raise ValueError(f"tile_size must be in [1, {TILE_SIZE}], got {tile_size}")
    n = r.shape[0]
    if n == 0:
        return ids[:0].copy(), np.empty(0, dtype=np.float32), ()
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    best_ids = np.empty(0, dtype=np.int64)
    best_d = np.empty(0, dtype=np.float32)
    tiles: list[int] = []
    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        tiles.append(end - start)
        d = exact_l2_distances(q, r[start:end])
        tile_ids = ids[start:end]
        tile_ids, tile_d = rank_topk(d, tile_ids, k)
        merged_ids = np.concatenate([best_ids, tile_ids])
        merged_d = np.concatenate([best_d, tile_d])
        merged_ids, merged_d = rank_topk(merged_d, merged_ids, k)
        best_ids, best_d = merged_ids, merged_d
    return best_ids, best_d, tuple(tiles)


def pack_raw_vector(vector: Any) -> bytes:
    v = np.asarray(vector, dtype=np.float32)
    if v.shape != (DIM,):
        raise LayoutError(f"raw vector must have shape ({DIM},), got {v.shape}")
    return v.tobytes()


def unpack_raw_vector(buf: bytes | bytearray | memoryview) -> np.ndarray:
    if len(buf) != RAW_VECTOR_BYTES:
        raise LayoutError(f"raw vector must be {RAW_VECTOR_BYTES} bytes, got {len(buf)}")
    return np.frombuffer(buf, dtype="<f4").copy()


def pack_pq_entry(code_row: Any, doc_id: int, doc_len: int = 0, flags: int = 0) -> bytes:
    codes = np.asarray(code_row, dtype=np.uint8)
    if codes.shape != (M,):
        raise LayoutError(f"code row must have shape ({M},), got {codes.shape}")
    if not 0 <= int(doc_id) < 2**64:
        raise LayoutError(f"doc_id {doc_id} does not fit in uint64")
    if not 0 <= int(doc_len) < 2**32 or not 0 <= int(flags) < 2**32:
        raise LayoutError("doc_len and flags must fit in uint32")
    meta = int(doc_id).to_bytes(8, "little") + int(doc_len).to_bytes(4, "little")
    meta += int(flags).to_bytes(4, "little")
    return codes.tobytes() + meta + b"\x00" * PAD_BYTES


def unpack_pq_entry(entry: bytes | bytearray | memoryview) -> tuple[np.ndarray, dict[str, int]]:
    if len(entry) != ENTRY_BYTES:
        raise LayoutError(f"PQ entry must be {ENTRY_BYTES} bytes, got {len(entry)}")
    codes = np.frombuffer(entry[:CODE_BYTES], dtype=np.uint8).copy()
    meta = bytes(entry[CODE_BYTES : CODE_BYTES + META_BYTES])
    doc_id = int.from_bytes(meta[0:8], "little")
    doc_len = int.from_bytes(meta[8:12], "little")
    flags = int.from_bytes(meta[12:16], "little")
    return codes, {"doc_id": doc_id, "doc_len": doc_len, "flags": flags}


def parse_cluster_header(header: bytes | bytearray | memoryview) -> dict[str, int]:
    """Parse the observed 64-byte R5 cluster header ``<III`` + 52 zero bytes."""
    if len(header) < 12:
        raise LayoutError(f"cluster header must be >= 12 bytes, got {len(header)}")
    m, dim, count = struct.unpack("<III", bytes(header[:12]))
    return {"m": m, "dim": dim, "count": count}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    for key in ("tenant", "index_version", "files"):
        if key not in manifest:
            raise LayoutError(f"manifest {path} is missing required key {key!r}")
    return manifest


def _load_array(entry: dict[str, Any], project_root: Path, verify_hash: bool) -> np.ndarray:
    raw_path = entry.get("path")
    if not raw_path:
        raise LayoutError(f"manifest entry has no path: {entry}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise LayoutError(f"manifest file {path} does not exist")
    dtype = np.dtype(entry.get("dtype", "float32"))
    shape = entry.get("shape")
    offset = int(entry.get("offset", 0))
    if offset < 0:
        raise LayoutError(f"manifest entry offset must be >= 0, got {offset}")
    if path.suffix == ".npy" or entry.get("format") == "npy":
        arr = np.load(path, allow_pickle=False)
    else:
        with open(path, "rb") as handle:
            handle.seek(offset)
            flat = np.fromfile(handle, dtype=dtype)
        arr = flat.reshape(tuple(shape)) if shape else flat
    if arr.dtype != dtype:
        raise LayoutError(f"{path} dtype {arr.dtype} != manifest dtype {dtype}")
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise LayoutError(f"{path} shape {arr.shape} != manifest shape {tuple(shape)}")
    if "nbytes" in entry and arr.nbytes != int(entry["nbytes"]):
        raise LayoutError(f"{path} nbytes {arr.nbytes} != manifest {entry['nbytes']}")
    if verify_hash and entry.get("sha256"):
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise LayoutError(f"{path} sha256 mismatch: {actual} != {entry['sha256']}")
    return arr


class IVFPQIndex:
    """Residual-L2 IVFPQ oracle over one tenant/index_version dataset."""

    def __init__(
        self,
        ref: DatasetRef,
        codebook: Any,
        centroids: Any,
        codes: Any,
        assign: Any,
        doc_ids: Any,
        raw: Any | None = None,
        queries: Any | None = None,
        groundtruth: Any | None = None,
    ) -> None:
        self.ref = ref
        self.codebook = np.asarray(codebook, dtype=np.float32)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.codes = np.asarray(codes, dtype=np.uint8)
        self.assign = np.asarray(assign, dtype=np.int64)
        self.doc_ids = np.asarray(doc_ids, dtype=np.int64)
        self.raw = None if raw is None else np.asarray(raw, dtype=np.float32)
        self.queries = None if queries is None else np.asarray(queries, dtype=np.float32)
        self.groundtruth = None if groundtruth is None else np.asarray(groundtruth)
        self.manifest: dict[str, Any] | None = None
        self._validate()
        self.cluster_rows = self._build_cluster_rows()
        self._doc_to_row = {int(d): i for i, d in enumerate(self.doc_ids.tolist())}

    def _validate(self) -> None:
        cb = self.codebook
        if cb.ndim != 3:
            raise LayoutError(f"codebook must be 3-D, got {cb.shape}")
        if cb.dtype != np.float32:
            raise LayoutError(f"codebook must be float32, got {cb.dtype}")
        m, ks, ds = cb.shape
        dim = m * ds
        if self.centroids.ndim != 2 or self.centroids.shape[1] != dim:
            raise LayoutError(f"centroids {self.centroids.shape} incompatible with dim {dim}")
        if self.codes.ndim != 2 or self.codes.shape[1] != m:
            raise LayoutError(f"codes {self.codes.shape} incompatible with M={m}")
        n = self.codes.shape[0]
        if self.assign.shape != (n,):
            raise LayoutError(f"assign {self.assign.shape} != ({n},)")
        if self.doc_ids.shape != (n,):
            raise LayoutError(f"doc_ids {self.doc_ids.shape} != ({n},)")
        if np.unique(self.doc_ids).shape[0] != n:
            raise LayoutError("doc_ids must be unique")
        nlist = self.centroids.shape[0]
        if n and (self.assign.min() < 0 or self.assign.max() >= nlist):
            raise LayoutError("assign contains cluster ids outside [0, nlist)")
        if self.raw is not None and self.raw.shape != (n, dim):
            raise LayoutError(f"raw {self.raw.shape} != ({n}, {dim})")
        if ks > 1 << NBITS:
            raise LayoutError(f"KS={ks} exceeds {NBITS}-bit codes")
        if self.queries is not None and self.queries.shape[1] != dim:
            raise LayoutError(f"queries {self.queries.shape} incompatible with dim {dim}")

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def n(self) -> int:
        return int(self.codes.shape[0])

    @property
    def nlist(self) -> int:
        return int(self.centroids.shape[0])

    def _build_cluster_rows(self) -> tuple[np.ndarray, ...]:
        nlist = self.nlist
        order = np.argsort(self.assign, kind="stable")
        counts = np.bincount(self.assign, minlength=nlist)
        rows: list[np.ndarray] = []
        start = 0
        for c in range(nlist):
            end = start + int(counts[c])
            rows.append(order[start:end])
            start = end
        return tuple(rows)

    def _check_ref(self, ref: DatasetRef | None) -> None:
        if ref is None:
            return
        if ref.tenant != self.ref.tenant or ref.index_version != self.ref.index_version:
            raise IdentityMismatchError(
                f"requested tenant={ref.tenant!r} index_version={ref.index_version!r} "
                f"but oracle holds tenant={self.ref.tenant!r} index_version={self.ref.index_version!r}"
            )
        if ref.cluster is not None and self.ref.cluster is not None and ref.cluster != self.ref.cluster:
            raise IdentityMismatchError(
                f"requested cluster={ref.cluster} but oracle holds cluster={self.ref.cluster}"
            )

    def _as_query(self, query: Any) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise LayoutError(f"query dim {q.shape[0]} != dataset dim {self.dim}")
        return q

    def _validate_r(self, r: int) -> int:
        r = int(r)
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        if r > MAX_R:
            raise ValueError(
                f"r={r} exceeds MAX_R={MAX_R}; refusing to silently truncate the candidate list"
            )
        return r

    def coarse_probes(self, query: Any, nprobe: int | None = None) -> np.ndarray:
        q = self._as_query(query)
        if nprobe is None:
            nprobe = min(DEFAULT_NPROBE, self.nlist)
        nprobe = int(nprobe)
        if not 1 <= nprobe <= self.nlist:
            raise ValueError(f"nprobe must be in [1, {self.nlist}], got {nprobe}")
        d = np.sum((self.centroids - q) ** 2, axis=1, dtype=np.float32)
        order = np.lexsort((np.arange(self.nlist, dtype=np.int64), d))
        return order[:nprobe]

    def cluster_candidates(
        self, query: Any, cluster: int, r: int = DEFAULT_R
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self._validate_r(r)
        q = self._as_query(query)
        cluster = int(cluster)
        if not 0 <= cluster < self.nlist:
            raise ValueError(f"cluster must be in [0, {self.nlist}), got {cluster}")
        rows = self.cluster_rows[cluster]
        if rows.shape[0] == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )
        table = pq_distance_table(residual_query(q, self.centroids[cluster]), self.codebook)
        d = pq_adc_distances(table, self.codes[rows])
        ids = self.doc_ids[rows]
        take = min(r, rows.shape[0])
        order = np.lexsort((ids, d))[:take]
        return ids[order], d[order], rows[order]

    def exact_topk(
        self,
        query: Any,
        k: int = DEFAULT_K,
        ref: DatasetRef | None = None,
        tile_size: int = TILE_SIZE,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        self._check_ref(ref)
        q = self._as_query(query)
        if self.n == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                (),
            )
        if self.raw is None:
            raise LayoutError("exact_topk requires raw base vectors")
        return scan_tiled(self.raw, self.doc_ids, q, k, tile_size)

    def search(
        self,
        query: Any,
        k: int = DEFAULT_K,
        nprobe: int | None = None,
        r: int = DEFAULT_R,
        ref: DatasetRef | None = None,
    ) -> SearchResult:
        self._check_ref(ref)
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        r = self._validate_r(r)
        q = self._as_query(query)
        probes = self.coarse_probes(q, nprobe)

        list_ids: list[np.ndarray] = []
        list_d: list[np.ndarray] = []
        list_rows: list[np.ndarray] = []
        for cluster in probes:
            ids, d, rows = self.cluster_candidates(q, int(cluster), r)
            if ids.shape[0]:
                list_ids.append(ids)
                list_d.append(d)
                list_rows.append(rows)
        if not list_ids:
            return SearchResult(
                doc_ids=np.empty(0, dtype=np.int64),
                distances=np.empty(0, dtype=np.float32),
                candidate_doc_ids=np.empty(0, dtype=np.int64),
                candidate_distances=np.empty(0, dtype=np.float32),
                candidate_rows=np.empty(0, dtype=np.int64),
            )

        cand_ids = np.concatenate(list_ids)
        cand_d = np.concatenate(list_d)
        cand_rows = np.concatenate(list_rows)
        cand_ids, unique_index = np.unique(cand_ids, return_index=True)
        cand_d = cand_d[unique_index]
        cand_rows = cand_rows[unique_index]

        if self.raw is None:
            raise LayoutError("search rerank requires raw base vectors")
        exact = exact_l2_distances(q, self.raw[cand_rows])
        top_ids, top_d = rank_topk(exact, cand_ids, min(int(k), cand_ids.shape[0]))
        return SearchResult(
            doc_ids=top_ids,
            distances=top_d,
            candidate_doc_ids=cand_ids,
            candidate_distances=cand_d,
            candidate_rows=cand_rows,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        tenant: str | None = None,
        index_version: str | None = None,
        project_root: str | Path | None = None,
        verify_hashes: bool = True,
    ) -> IVFPQIndex:
        manifest = load_manifest(manifest_path)
        if tenant is not None and tenant != manifest["tenant"]:
            raise IdentityMismatchError(
                f"requested tenant={tenant!r} but manifest tenant={manifest['tenant']!r}"
            )
        if index_version is not None and index_version != manifest["index_version"]:
            raise IdentityMismatchError(
                f"requested index_version={index_version!r} "
                f"but manifest index_version={manifest['index_version']!r}"
            )
        root = Path(project_root) if project_root is not None else DEFAULT_PROJECT_ROOT
        files = manifest["files"]

        index = cls(
            ref=DatasetRef(
                tenant=manifest["tenant"],
                index_version=manifest["index_version"],
                cluster=manifest.get("cluster"),
            ),
            codebook=_load_array(files["codebook"], root, verify_hashes),
            centroids=_load_array(files["centroids"], root, verify_hashes),
            codes=_load_array(files["codes"], root, verify_hashes),
            assign=_load_array(files["assign"], root, verify_hashes),
            doc_ids=_load_array(files["doc_ids"], root, verify_hashes),
            raw=_load_array(files["raw"], root, verify_hashes) if "raw" in files else None,
            queries=_load_array(files["queries"], root, verify_hashes) if "queries" in files else None,
            groundtruth=(
                _load_array(files["groundtruth"], root, verify_hashes)
                if "groundtruth" in files
                else None
            ),
        )
        index.manifest = manifest
        return index
