"""Bounded, single-owner query-global approximate-IP candidate merger.

Only finish() publishes results, after the caller closes the complete input
stream. Each required list must arrive exactly once. Any invalid input, reported
list error, or premature finish permanently poisons this owner. No retry/refill,
raw-vector computation, qrels, execution timing, or cost authorization is provided.

Exactness is conditional on producers supplying genuine local top-depth results
under (-approximate_IP, row_id), and on the pinned ownership declaration describing
the actual index. Digests bind declarations here; this helper cannot verify index
bytes or unseen scores. Exact top-R is over the required lists' approximate scores,
not full-precision IP or unprobed lists. Ownership uses either disjoint half-open
uint64 ranges (declarations only) or a hash-verified explicit membership manifest.
"""
from dataclasses import dataclass
import hashlib
import heapq
import math
import re
from typing import Optional, Tuple
from types import MappingProxyType


UINT64_MAX = (1 << 64) - 1
MAX_LISTS = 65536
MAX_CANDIDATES = 1000000
MAX_OWNERSHIP_ENTRIES = 1000000
MAX_OWNERSHIP_BYTES = 16 * 1024 * 1024
_MEMBERSHIP_MAGIC = b"cortex.membership-ownership.v1\0"


class MergeError(ValueError):
    """Invalid configuration or permanently failed query barrier."""


def _require(condition, message):
    if not condition:
        raise MergeError(message)


def _uint(value, label, minimum=0, maximum=UINT64_MAX):
    _require(type(value) is int and minimum <= value <= maximum, "invalid " + label)


def _text(value):
    return type(value) is str and bool(value) and value == value.strip()


@dataclass(frozen=True)
class QueryIdentity:
    index_namespace: str
    index_version: str
    index_sha256: str
    query_id: str
    query_sha256: str
    document_vectors_sha256: str
    ownership_sha256: str
    profile_sha256: str
    encoder_revision: str
    metric: str
    top_r: int
    top_k: int
    nprobe: int
    by_residual: bool
    # Canonical, sorted, unique text pairs; pin all producer-specific search knobs.
    search_config: Tuple[Tuple[str, str], ...]


def _identity(value):
    _require(type(value) is QueryIdentity, "invalid query identity")
    for name in ("index_namespace", "index_version", "query_id"):
        _require(_text(getattr(value, name)), "invalid identity text")
    for name in ("index_sha256", "query_sha256", "document_vectors_sha256",
                 "ownership_sha256", "profile_sha256", "encoder_revision"):
        size = 40 if name == "encoder_revision" else 64
        digest = getattr(value, name)
        _require(type(digest) is str and re.fullmatch("[0-9a-f]{%d}" % size, digest),
                 "invalid identity hash: " + name)
    for name in ("top_r", "top_k", "nprobe"):
        _uint(getattr(value, name), name, minimum=1)
    _require(value.top_k <= value.top_r, "top_k exceeds top_r")
    _require(type(value.metric) is str and value.metric == "IP"
             and type(value.by_residual) is bool, "explicit IP/residual identity required")
    config = value.search_config
    _require(type(config) is tuple and bool(config), "explicit immutable search configuration required")
    previous = None
    for pair in config:
        _require(type(pair) is tuple and len(pair) == 2 and all(_text(v) for v in pair),
                 "invalid search configuration pair")
        _require(previous is None or previous < pair[0], "search keys must be sorted and unique")
        previous = pair[0]


@dataclass(frozen=True)
class ListSpec:
    list_id: str
    row_start: int
    row_stop: int
    cardinality: int


@dataclass(frozen=True)
class MembershipSpec:
    """Input rows may be list/tuple; startup takes a validated sorted tuple copy."""

    list_id: str
    cardinality: int
    rows: Tuple[int, ...]


def _membership(required_lists, max_lists, max_entries, max_bytes):
    """Preflight sizes before row copies; return frozen content, owners and SHA256.

    Binary canonical form: magic, u64 list count; per list in manifest order,
    u64 UTF-8 ID byte length, ID bytes, u64 cardinality, ascending u64 rows.
    All integers are unsigned big-endian. Encoded-byte budget includes framing.
    """
    _uint(max_lists, "max_lists", 1, MAX_LISTS)
    _uint(max_entries, "max_ownership_entries", 0, MAX_OWNERSHIP_ENTRIES)
    _uint(max_bytes, "max_ownership_bytes", 1, MAX_OWNERSHIP_BYTES)
    _require(type(required_lists) is tuple and 0 < len(required_lists) <= max_lists,
             "invalid required ordered lists or max_lists exceeded")
    entries = 0
    encoded = len(_MEMBERSHIP_MAGIC) + 8
    names = set()
    for spec in required_lists:
        _require(type(spec) is MembershipSpec and _text(spec.list_id),
                 "invalid membership specification")
        _require(spec.list_id not in names, "duplicate required list")
        names.add(spec.list_id)
        _uint(spec.cardinality, "source cardinality")
        _require(type(spec.rows) in (tuple, list) and len(spec.rows) == spec.cardinality,
                 "membership cardinality mismatch")
        _require(spec.cardinality <= max_entries - entries, "ownership entry budget exceeded")
        entries += spec.cardinality
        # UTF-8 has at least one byte per character; reject oversized IDs before encoding.
        _require(len(spec.list_id) <= max_bytes - encoded, "ownership byte budget exceeded")
        try:
            name_size = len(spec.list_id.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise MergeError("invalid UTF-8 list ID") from exc
        size = 16 + name_size + 8 * spec.cardinality
        _require(size <= max_bytes - encoded, "ownership byte budget exceeded")
        encoded += size
    _require(encoded <= max_bytes, "ownership byte budget exceeded")
    # Every count/list identity is now checked; validate rows before freezing content.
    owners = {}
    for spec in required_lists:
        for row in spec.rows:
            _uint(row, "ownership row ID")
            _require(row not in owners, "duplicate ownership row")
            owners[row] = spec.list_id
    frozen = tuple(MembershipSpec(s.list_id, s.cardinality, tuple(sorted(s.rows)))
                   for s in required_lists)
    digest = hashlib.sha256(_MEMBERSHIP_MAGIC)
    digest.update(len(frozen).to_bytes(8, "big"))
    for spec in frozen:
        name = spec.list_id.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(spec.cardinality.to_bytes(8, "big"))
        for row in spec.rows:
            digest.update(row.to_bytes(8, "big"))
    return frozen, MappingProxyType(owners), digest.hexdigest()


def membership_ownership_sha256(required_lists, *, max_lists=MAX_LISTS,
                                max_ownership_entries=MAX_OWNERSHIP_ENTRIES,
                                max_ownership_bytes=MAX_OWNERSHIP_BYTES):
    """Compute validated canonical content hash; trust in that digest is external.

    Merger construction independently recomputes and compares the expected digest.
    Callers must not derive the trusted expectation from untrusted arrivals.
    """
    return _membership(required_lists, max_lists, max_ownership_entries,
                       max_ownership_bytes)[2]


@dataclass(frozen=True)
class Candidate:
    row_id: int
    score: float


@dataclass(frozen=True)
class ListResult:
    identity: QueryIdentity
    list_id: str
    cardinality: int
    local_depth: int
    candidates: Tuple[Candidate, ...]
    error: Optional[str] = None


class GlobalCandidateMerger:
    """One query, no concurrent calls; consume() never returns candidates.

    required_lists is an ordered immutable manifest, with exactly nprobe unique
    IDs. Use exclusively ListSpec ranges or MembershipSpec explicit rows, never
    mixed modes. Membership inputs are copied/frozen after bounded validation;
    callers must not mutate inputs during construction. Their canonical content
    hash must equal the independently trusted identity.ownership_sha256.
    R candidates per list suffice only because ownership is disjoint.
    Depth must equal supplied length and cover min(R, cardinality); greater depth
    is allowed within cumulative max_candidates. Heap retention is at most R;
    transient validation retains one local ID set, never caller-owned containers.
    finish() seals the stream (and is idempotent); no later consume is permitted.
    A caller must not publish until all producer outcomes have been delivered and
    finish succeeds. Previously returned immutable tuples cannot be revoked.
    """

    def __init__(self, identity, required_lists, *, max_lists=MAX_LISTS,
                 max_candidates=MAX_CANDIDATES,
                 max_ownership_entries=MAX_OWNERSHIP_ENTRIES,
                 max_ownership_bytes=MAX_OWNERSHIP_BYTES):
        _identity(identity)
        _uint(max_lists, "max_lists", 1, MAX_LISTS)
        _uint(max_candidates, "max_candidates", 1, MAX_CANDIDATES)
        _require(type(required_lists) is tuple and 0 < len(required_lists) <= max_lists,
                 "invalid required ordered lists or max_lists exceeded")
        _require(identity.nprobe == len(required_lists), "nprobe/list count mismatch")
        _require(identity.top_r <= max_candidates, "top_r exceeds candidate budget")
        _uint(max_ownership_entries, "max_ownership_entries", 0, MAX_OWNERSHIP_ENTRIES)
        _uint(max_ownership_bytes, "max_ownership_bytes", 1, MAX_OWNERSHIP_BYTES)
        owners = None
        if type(required_lists[0]) is MembershipSpec:
            required_lists, owners, digest = _membership(
                required_lists, max_lists, max_ownership_entries, max_ownership_bytes)
            _require(digest == identity.ownership_sha256, "ownership digest mismatch")
        specs = {}
        cardinality = minimum = 0
        for spec in required_lists:
            expected_type = MembershipSpec if owners is not None else ListSpec
            _require(type(spec) is expected_type and _text(spec.list_id), "invalid list specification")
            _require(spec.list_id not in specs, "duplicate required list")
            _uint(spec.cardinality, "source cardinality")
            if owners is None:
                _uint(spec.row_start, "ownership start")
                _uint(spec.row_stop, "ownership stop", 1, UINT64_MAX + 1)
                _require(spec.row_start < spec.row_stop
                         and spec.cardinality <= spec.row_stop - spec.row_start,
                         "invalid ownership range/cardinality")
            _require(cardinality <= UINT64_MAX - spec.cardinality, "cardinality sum overflow")
            cardinality += spec.cardinality
            depth = min(identity.top_r, spec.cardinality)
            _require(minimum <= max_candidates - depth, "minimum candidates exceed budget")
            minimum += depth
            specs[spec.list_id] = spec
        if owners is None:
            ranges = sorted((s.row_start, s.row_stop) for s in required_lists)
            _require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])),
                     "disjoint document ownership required")
        self._identity = identity
        self._required_lists = required_lists
        self._specs = specs
        self._owners = owners
        self._max_candidates = max_candidates
        self._cardinality = cardinality
        self._received = set()
        self._candidate_count = 0
        self._heap = []
        self._result = None
        self._state = "accepting"

    @property
    def state(self):
        return self._state

    @property
    def consumed_count(self):
        return len(self._received)

    @property
    def identity(self):
        return self._identity

    @property
    def required_lists(self):
        return self._required_lists

    def _fail(self):
        self._state = "failed"
        self._heap.clear()
        self._result = None

    def consume(self, result):
        """Validate an entire packet before marking its list consumed; return None."""
        try:
            _require(self._state == "accepting", "query is closed or failed")
            _require(type(result) is ListResult, "invalid list result")
            _identity(result.identity)
            _require(result.identity == self._identity, "full query identity mismatch")
            _require(_text(result.list_id) and result.list_id in self._specs, "unexpected list")
            _require(result.list_id not in self._received, "duplicate list arrival")
            _require(result.error is None, "list error")
            spec = self._specs[result.list_id]
            _uint(result.cardinality, "source cardinality")
            _uint(result.local_depth, "local truncation depth")
            _require(result.cardinality == spec.cardinality, "source cardinality mismatch")
            _require(min(self._identity.top_r, spec.cardinality) <= result.local_depth
                     <= spec.cardinality, "insufficient or invalid local depth")
            _require(result.local_depth <= self._max_candidates - self._candidate_count,
                     "candidate budget exceeded")
            _require(type(result.candidates) is tuple and len(result.candidates) == result.local_depth,
                     "candidate count/declared depth mismatch")
            seen = set()
            previous = None
            for candidate in result.candidates:
                _require(type(candidate) is Candidate, "invalid candidate")
                _uint(candidate.row_id, "document row ID")
                _require(type(candidate.score) in (int, float), "invalid score")
                try:
                    finite = math.isfinite(candidate.score)
                except OverflowError:
                    finite = False
                _require(finite, "nonfinite or unrepresentable approximate IP")
                belongs = (self._owners.get(candidate.row_id) == spec.list_id
                           if self._owners is not None else
                           spec.row_start <= candidate.row_id < spec.row_stop)
                _require(belongs,
                         "document outside disjoint ownership")
                _require(candidate.row_id not in seen, "duplicate local document")
                key = (-candidate.score, candidate.row_id)
                _require(previous is None or previous < key, "invalid local total ordering")
                seen.add(candidate.row_id)
                previous = key
            # All identity/count/order/ownership checks precede barrier mutation.
            for candidate in result.candidates:
                entry = (candidate.score, -candidate.row_id, candidate.row_id)
                if len(self._heap) < self._identity.top_r:
                    heapq.heappush(self._heap, entry)
                elif entry > self._heap[0]:
                    heapq.heapreplace(self._heap, entry)
            self._candidate_count += result.local_depth
            self._received.add(result.list_id)
        except MergeError:
            self._fail()
            raise

    def finish(self):
        """Close all inputs; missing lists poison the query, never publish a prefix."""
        try:
            _require(self._state != "failed", "query failed")
            if self._state == "complete":
                return self._result
            _require(len(self._received) == len(self._required_lists), "missing required list results")
            _require(len(self._heap) == min(self._identity.top_r, self._cardinality),
                     "incomplete global coverage")
            self._result = tuple(Candidate(row, score) for score, negrow, row
                                 in sorted(self._heap, reverse=True))
            self._heap.clear()
            self._state = "complete"
            return self._result
        except MergeError:
            self._fail()
            raise
