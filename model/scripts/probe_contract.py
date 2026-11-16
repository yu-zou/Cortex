#!/usr/bin/env python3
"""Immutable BEIR probe contract, independent of scheduler/runtime code.

Centroid IDs are local to an explicitly pinned namespace and index version.
Only the selected nprobe centroids occur in each row; cluster_sizes is the
complete, externally supplied size table. Probe order is retained verbatim.
Candidate counts describe the union of selected inverted lists, not doc IDs
or retrieval results. No retrieval quality or FPGA execution is implied.
"""
import argparse
import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple


MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
MPNET_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def _text(name, value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("{} must be a nonempty, unpadded string".format(name))


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))


@dataclass(frozen=True)
class ClusterIdentity:
    index_namespace: str
    index_version: str
    cluster_local_id: int

    def __post_init__(self):
        _text("index_namespace", self.index_namespace)
        _text("index_version", self.index_version)
        _integer("cluster_local_id", self.cluster_local_id)

    @property
    def key(self):
        """Structured key; never delimiter-join or process-random hash IDs."""
        return (self.index_namespace, self.index_version, self.cluster_local_id)


@dataclass(frozen=True)
class IndexMetadata:
    index_namespace: str
    index_version: str
    dataset: str
    index_id: str
    encoder_revision: str
    dimension: int
    metric: str
    normalized: bool
    split: str
    total_centroids: int
    nprobe: int
    encoder_model: str = MPNET_MODEL

    def __post_init__(self):
        for name in ("index_namespace", "index_version", "dataset", "index_id",
                     "encoder_revision", "metric", "split", "encoder_model"):
            _text(name, getattr(self, name))
        if not re.fullmatch(r"[0-9a-f]{40}", self.encoder_revision):
            raise ValueError("encoder_revision must be a pinned 40-hex commit")
        for name in ("dimension", "total_centroids", "nprobe"):
            _integer(name, getattr(self, name), 1)
        if self.encoder_model == MPNET_MODEL and self.dimension != 768:
            raise ValueError("all-mpnet-base-v2 dimension must be 768")
        if self.metric not in ("IP", "L2"):
            raise ValueError("metric must be IP or L2; declare normalization separately")
        if type(self.normalized) is not bool:
            raise ValueError("normalized must be boolean")
        if self.nprobe > self.total_centroids:
            raise ValueError("nprobe exceeds total centroid cardinality")


@dataclass(frozen=True)
class Probe:
    metadata: IndexMetadata
    query_id: str
    centroid_ids: Tuple[int, ...]
    cluster_sizes: Tuple[int, ...]
    candidate_count: int
    query_row: Optional[int] = None

    @property
    def cluster_identities(self):
        return tuple(ClusterIdentity(self.metadata.index_namespace,
                                     self.metadata.index_version, cid)
                     for cid in self.centroid_ids)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def strict_json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def load_probes(source, metadata, cluster_sizes):
    """Load a path or text stream atomically, returning an immutable tuple.

    Metadata is never inferred from a filename or accepted from query rows.
    The full size table validates local IDs even when nprobe < nlist.
    """
    if not isinstance(metadata, IndexMetadata):
        raise ValueError("explicit IndexMetadata required")
    sizes = tuple(cluster_sizes)
    if len(sizes) != metadata.total_centroids:
        raise ValueError("cluster size table cardinality mismatch")
    for size in sizes:
        _integer("cluster size table entry", size)
    if hasattr(source, "read"):
        return _load_rows(source, metadata, sizes)
    with open(source, encoding="utf-8") as stream:
        return _load_rows(stream, metadata, sizes)


def _load_rows(stream, metadata, sizes):
    required = {"query_id", "centroid_ids", "cluster_sizes", "candidate_count"}
    optional = {"query_row", "nprobe"}
    queries = set()
    query_rows = set()
    probes = []
    for line_no, line in enumerate(stream, 1):
        try:
            row = strict_json(line)
            if not isinstance(row, dict) or not required <= row.keys() or row.keys() - required - optional:
                raise ValueError("missing or unknown probe fields")
            qid = row["query_id"]
            _text("query_id", qid)
            if qid in queries:
                raise ValueError("duplicate query: " + qid)
            if "query_row" in row:
                _integer("query_row", row["query_row"])
                if row["query_row"] in query_rows:
                    raise ValueError("duplicate query_row")
                query_rows.add(row["query_row"])
            if "nprobe" in row:
                _integer("nprobe", row["nprobe"], 1)
                if row["nprobe"] != metadata.nprobe:
                    raise ValueError("row nprobe differs from metadata")
            ids, counts = row["centroid_ids"], row["cluster_sizes"]
            if not isinstance(ids, list) or not isinstance(counts, list):
                raise ValueError("centroid_ids and cluster_sizes must be arrays")
            if len(ids) != metadata.nprobe or len(counts) != len(ids):
                raise ValueError("selected probe cardinality mismatch")
            for cid, size in zip(ids, counts):
                _integer("centroid ID", cid)
                _integer("cluster size", size)
                if cid >= metadata.total_centroids:
                    raise ValueError("centroid ID out of range")
                if size != sizes[cid]:
                    raise ValueError("cluster size differs from full size table")
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate clusters")
            _integer("candidate_count", row["candidate_count"])
            if row["candidate_count"] != sum(counts):
                raise ValueError("candidate_count differs from sum of selected cluster sizes")
            probes.append(Probe(metadata, qid, tuple(ids), tuple(counts),
                                row["candidate_count"], row.get("query_row")))
            queries.add(qid)
        except (ValueError, TypeError) as exc:
            raise ValueError("line {}: {}".format(line_no, exc)) from exc
    if not probes:
        raise ValueError("empty probe input")
    return tuple(probes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--cluster-sizes", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    args = parser.parse_args()
    try:
        def read_json(path):
            with open(path, encoding="utf-8") as stream:
                return strict_json(stream.read())
        metadata = IndexMetadata(**read_json(args.metadata))
        encoder = read_json(args.encoder_config)
        for field, expected in (("model", metadata.encoder_model),
                                ("revision", metadata.encoder_revision),
                                ("dimension", metadata.dimension),
                                ("normalize_embeddings", metadata.normalized)):
            if type(encoder.get(field)) is not type(expected) or encoder[field] != expected:
                raise ValueError("encoder_config mismatch: " + field)
        probes = load_probes(args.probes, metadata, read_json(args.cluster_sizes))
        if args.expected_queries <= 0 or len(probes) != args.expected_queries:
            raise ValueError("unexpected query count")
        print(json.dumps({"query_count": len(probes), "metadata": metadata.__dict__,
                          "query_id_mapping": [{"row": i, "query_id": p.query_id,
                                                "query_row": p.query_row} for i, p in enumerate(probes)],
                          "total_centroids": metadata.total_centroids,
                          "probes_per_query": metadata.nprobe}, indent=2))
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(1, "probe contract: {}\n".format(exc))


if __name__ == "__main__":
    main()
