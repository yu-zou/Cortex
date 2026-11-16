"""Strict, offline request accounting in one explicitly declared clock domain.

Times are nonnegative integer nanoseconds. Send time defines system arrival and
end-to-end latency. Planned time is scheduled demand, not admission. The caller
supplies the independent sent-request ID manifest: completeness cannot be
inferred from completion logs alone. Unsent censored records are allowed but
are not admitted requests and must not appear in that manifest.
The observation horizon is inclusive; windows are half open. Backend/profile
labels are provenance only, never a substitute for clock-domain declarations.
"""
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(name + " must be a nonempty unpadded string")


def _integer(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")


def _domain(value: str) -> None:
    if value not in ("client_monotonic", "des_logical"):
        raise ValueError("time_domain must be client_monotonic or des_logical")


@dataclass(frozen=True)
class RunMetadata:
    backend_label: str
    profile: str
    index: str
    version: str

    def __post_init__(self) -> None:
        for name in ("backend_label", "profile", "index", "version"):
            _text(name, getattr(self, name))


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    time_domain: str
    planned_ns: int
    send_ns: Optional[int]
    terminal_ns: Optional[int]
    status: str
    bytes_in: int
    bytes_out: int
    metadata: RunMetadata

    def __post_init__(self) -> None:
        _text("request_id", self.request_id)
        _domain(self.time_domain)
        for name in ("planned_ns", "bytes_in", "bytes_out"):
            _integer(name, getattr(self, name))
        if not isinstance(self.metadata, RunMetadata):
            raise ValueError("explicit RunMetadata required")
        if self.status not in ("ok", "error", "timeout", "censored"):
            raise ValueError("invalid request status")
        if self.send_ns is not None:
            _integer("send_ns", self.send_ns)
            if self.send_ns < self.planned_ns:
                raise ValueError("send precedes planned arrival")
        if self.status == "censored":
            if self.terminal_ns is not None:
                raise ValueError("censored request must not have a terminal timestamp")
        else:
            _integer("terminal_ns", self.terminal_ns)
            if self.send_ns is None or self.terminal_ns < self.send_ns:
                raise ValueError("terminal request needs send <= terminal")


@dataclass(frozen=True)
class CostRecord:
    build: float
    run: float
    storage: float
    currency: str

    def __post_init__(self) -> None:
        for name in ("build", "run", "storage"):
            value = getattr(self, name)
            try:
                valid = (type(value) in (int, float) and value >= 0
                         and math.isfinite(value))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(name + " cost must be finite and nonnegative")
        if not isinstance(self.currency, str) or not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise ValueError("currency must be an explicit three-uppercase-letter code")


def _percentile_ms(values: Iterable[int], percent: int) -> Optional[float]:
    """Linear interpolation, matching the existing helper's percentile rule."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return (ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)) / 1e6


@dataclass(frozen=True)
class Measurement:
    """One uniform run; arrived_request_ids is the independent sent-ID manifest."""

    requests: Tuple[RequestRecord, ...]
    arrived_request_ids: Tuple[str, ...]
    time_domain: str
    observation_horizon_ns: int

    def __post_init__(self) -> None:
        _domain(self.time_domain)
        _integer("observation_horizon_ns", self.observation_horizon_ns)
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(self, "arrived_request_ids", tuple(self.arrived_request_ids))
        ids = set()
        admitted_ids = set()
        for request in self.requests:
            if not isinstance(request, RequestRecord):
                raise ValueError("explicit RequestRecord required")
            if request.request_id in ids:
                raise ValueError("duplicate request ID: " + request.request_id)
            ids.add(request.request_id)
            if request.send_ns is not None:
                admitted_ids.add(request.request_id)
            if request.metadata != self.requests[0].metadata:
                raise ValueError("mixed run metadata (backend/profile/index/version)")
            if request.time_domain != self.time_domain:
                raise ValueError("mixed clock domains")
            for timestamp in (request.planned_ns, request.send_ns, request.terminal_ns):
                if timestamp is not None and timestamp > self.observation_horizon_ns:
                    raise ValueError("request timestamp exceeds observation horizon")
        for request_id in self.arrived_request_ids:
            _text("arrived request ID", request_id)
        if len(set(self.arrived_request_ids)) != len(self.arrived_request_ids):
            raise ValueError("duplicate arrived request ID")
        if admitted_ids != set(self.arrived_request_ids):
            raise ValueError("every sent request must have exactly one arrived record")

    def _backlog(self, t: int) -> int:
        return sum(r.send_ns is not None and r.send_ns < t
                   and (r.terminal_ns is None or t <= r.terminal_ns)
                   for r in self.requests)

    def window(self, start_ns: int, end_ns: int, *,
               boundaries_ns: Optional[Iterable[int]] = None) -> Dict[str, Any]:
        """Count all arrivals/exits, with backlog at each selected boundary.

        Default boundaries include window endpoints and every observed event
        inside the window. Explicit boundaries may span the entire observation.
        Sent censored requests remain outstanding through the horizon; unsent
        requests never enter system backlog or the send-time cohort. Successful
        latency is terminal minus send; send delay is send minus planned for all
        cohort members. Extra terminal-minus-planned statistics are explicitly
        named and describe only successful cohort members. Complete-cohort
        percentiles exist only when every admitted member is observed successful.
        """
        _integer("start_ns", start_ns)
        _integer("end_ns", end_ns)
        if end_ns <= start_ns or end_ns > self.observation_horizon_ns:
            raise ValueError("window must be nonempty and within observation horizon")
        if boundaries_ns is None:
            boundaries = tuple(sorted({start_ns, end_ns} | {
                t for r in self.requests for t in (r.send_ns, r.terminal_ns)
                if t is not None and start_ns <= t <= end_ns}))
        else:
            boundaries = tuple(boundaries_ns)
        previous = -1
        for t in boundaries:
            _integer("boundary", t)
            if t <= previous or t > self.observation_horizon_ns:
                raise ValueError("boundaries must increase strictly within observation horizon")
            previous = t
        cohort = [r for r in self.requests
                  if r.send_ns is not None and start_ns <= r.send_ns < end_ns]
        exits = [r for r in self.requests
                 if r.terminal_ns is not None and start_ns <= r.terminal_ns < end_ns]
        ok = sum(r.status == "ok" for r in exits)
        errors = sum(r.status == "error" for r in exits)
        timeouts = sum(r.status == "timeout" for r in exits)
        successful = [r.terminal_ns - r.send_ns for r in cohort if r.status == "ok"]
        send_delays = [r.send_ns - r.planned_ns for r in cohort]
        scheduled_response = [r.terminal_ns - r.planned_ns for r in cohort if r.status == "ok"]
        complete = bool(cohort) and len(successful) == len(cohort)
        b0, b1 = self._backlog(start_ns), self._backlog(end_ns)
        return dict(
            time_domain=self.time_domain, observation_horizon_ns=self.observation_horizon_ns,
            metadata=asdict(self.requests[0].metadata) if self.requests else None,
            start_ns=start_ns, end_ns=end_ns, arrival_count=len(cohort),
            ok_completion_count=ok, completion_qps=ok * 1e9 / (end_ns - start_ns),
            terminal_failure_count=errors + timeouts, error_count=errors, timeout_count=timeouts,
            outstanding_start=b0, outstanding_end=b1, outstanding_growth=b1 - b0,
            conservation_residual=len(cohort) - ok - errors - timeouts - (b1 - b0),
            backlog=tuple((t, self._backlog(t)) for t in boundaries),
            cohort_count=len(cohort), cohort_ok_count=len(successful),
            cohort_failure_count=sum(r.status in ("error", "timeout") for r in cohort),
            cohort_censored_count=sum(r.status == "censored" for r in cohort),
            successful_latency_p50_ms=_percentile_ms(successful, 50),
            successful_latency_p99_ms=_percentile_ms(successful, 99),
            send_delay_p50_ms=_percentile_ms(send_delays, 50),
            send_delay_p99_ms=_percentile_ms(send_delays, 99),
            successful_terminal_minus_planned_p50_ms=_percentile_ms(scheduled_response, 50),
            successful_terminal_minus_planned_p99_ms=_percentile_ms(scheduled_response, 99),
            cohort_complete_p50_ms=_percentile_ms(successful, 50) if complete else None,
            cohort_complete_p99_ms=_percentile_ms(successful, 99) if complete else None)
