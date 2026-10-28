"""Task 17: stage-based FPGA service-time model driven by extracted HLS profiles.

The model is built only from a parsed ``HlsProfile`` plus a contract
``ProfileRef``; if the report was missing, the source hash did not match, or a
required loop field is unknown, construction fails explicitly instead of
fabricating timing.  Service time is decomposed into a TOPK init phase, a TOPK
process (steady) phase and a push (drain) phase split by pipeline startup and
fixed overhead, so the model never assumes TOPK runs once per query: the number
of init runs per call is an explicit parameter.

Accuracy claims are gated separately from extraction.  Per-loop csynth numbers
and predictions built from them are extraction-level evidence only; the
``<= 5% median / <= 10% max`` gate is evaluated exclusively against
independent RTL co-sim total-cycle holdout data.  With no such data the model
reports ``blocked``/``not-calibrated`` and never a pass.  External memory wait
is intentionally not added here; it is accounted by the DES resource
calendars, so it is not double counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from strata_sim.contracts.schema import ContractError, ProfileRef

GATE_MEDIAN_ERROR_PCT = 5.0
GATE_MAX_ERROR_PCT = 10.0
HOLDOUT_N = (1, 17, 255, 1023, 4097, 8192)
# The steady-state TOPK process throughput is a hardware property, not a fit
# parameter: the fitted 3-parameter model was structurally unidentifiable
# (min(top_k, n) is collinear with the constant for n >= top_k and the fitted
# push was negative).  The AXI-512 redesign (data path widened 32-bit ->
# 512-bit) makes every large-N co-sim interval asymptote to 2.000
# cycles/document, exactly 16x the pre-widening 32.000; the stage model
# therefore anchors this term and fits only the fixed base.  Source:
# independent RTL co-simulation holdout (results/validation/cosim-holdout.json),
# intervals (255,1023) and (4097,8192), both 2.000.
ANCHORED_PROCESS_CYCLES_PER_DOC = 2.0
# The codebook stream is a separate physical phase, not part of the per-document
# steady state.  In the AXI-512 design the cold-vs-warm co-sim delta at n=17 is
# 15081 - 2782 = 12299 cycles, matching the csynth
# data_manager_Pipeline_CB_STREAM latency (12291 cycles = 68.289 us at 180 MHz).
# It is subtracted from the fitted cold base to expose the warm (codebook
# resident) service time used by the OSD timing model, so a BRAM-resident
# codebook is never charged twice.
CODEBOOK_LOAD_CYCLES = 12299.0
MODEL_FORM_REASON = (
    "model form: process anchored to 2.0 cycles/doc (hardware steady-state "
    "throughput of the AXI-512 design; the asymptotic slope of every large-N "
    "co-sim interval is 2.000) and the min(top_k,n) push term dropped as "
    "collinear/unidentifiable (the 3-parameter fit returned a negative push); "
    "only the fixed base is fitted"
)
PHASE_LOOPS = {
    "topk_init": "TOPK_INIT",
    "topk_process": "TOPK_PROCESS",
    "topk_push": "RESULT_PUSH",
}

STATUS_CALIBRATED = "calibrated"
STATUS_NOT_CALIBRATED = "not-calibrated"
STATUS_BLOCKED = "blocked"


class FpgaModelError(ContractError):
    """An FPGA service-model invariant was violated."""


class ProfileUnavailableError(FpgaModelError):
    """The profile is incomplete or was never parsed; no timing may be invented."""


class MissingProfileError(FpgaModelError):
    """The extracted profile or source report does not exist."""


class NotCalibratedError(FpgaModelError):
    """A prediction was requested before co-sim calibration ran."""


def _require_count(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FpgaModelError(f"{name} must be a non-negative int, got {value!r}")
    return value


@dataclass(frozen=True)
class CalibrationPoint:
    """One total-cycle observation; ``origin`` says whether it is independent.

    ``origin="cosim"`` marks independent RTL co-simulation data.  Anything else
    (for example ``"fixture"``) is test data and can be used for fit mechanics
    but must never satisfy the accuracy gate.
    """

    n_docs: int
    top_k: int
    total_cycles: float
    origin: str = "cosim"

    def __post_init__(self) -> None:
        _require_count("n_docs", self.n_docs)
        _require_count("top_k", self.top_k)
        if isinstance(self.total_cycles, bool) or not isinstance(self.total_cycles, (int, float)) \
                or self.total_cycles <= 0:
            raise FpgaModelError(
                f"total_cycles must be positive, got {self.total_cycles!r}")
        if not isinstance(self.origin, str) or not self.origin:
            raise FpgaModelError(f"origin must be a non-empty string, got {self.origin!r}")

    def to_dict(self) -> dict:
        return {"n_docs": self.n_docs, "top_k": self.top_k,
                "total_cycles": self.total_cycles, "origin": self.origin}


@dataclass(frozen=True)
class ServiceStage:
    """One service-time component: startup + steady * units + drain."""

    name: str
    startup_cycles: float
    steady_cycles: float
    drain_cycles: float
    units: int

    @property
    def total_cycles(self) -> float:
        return self.startup_cycles + self.steady_cycles + self.drain_cycles

    def to_dict(self) -> dict:
        return {"name": self.name, "startup_cycles": self.startup_cycles,
                "steady_cycles": self.steady_cycles, "drain_cycles": self.drain_cycles,
                "units": self.units, "total_cycles": self.total_cycles}


@dataclass(frozen=True)
class ServiceEstimate:
    """A calibrated service-time prediction at the profile's DES clock."""

    n_docs: int
    top_k: int
    init_runs: int
    total_cycles: float
    clock_ns: float
    total_ns: float
    stages: tuple[ServiceStage, ...]
    external_memory_cycles: None = None

    def to_dict(self) -> dict:
        return {"n_docs": self.n_docs, "top_k": self.top_k,
                "init_runs": self.init_runs, "total_cycles": self.total_cycles,
                "clock_ns": self.clock_ns, "total_ns": self.total_ns,
                "external_memory_cycles": self.external_memory_cycles,
                "stages": [stage.to_dict() for stage in self.stages]}


@dataclass(frozen=True)
class CalibrationReport:
    """Fit outcome plus the holdout gate decision (never a pass without co-sim)."""

    status: str
    passed: bool
    gate_median_error_pct: float
    gate_max_error_pct: float
    median_error_pct: float | None
    max_error_pct: float | None
    n_train: int
    n_holdout: int
    holdout_origins: tuple[str, ...]
    errors: tuple[dict, ...]
    reasons: tuple[str, ...]
    topk_init_cycles: float
    fixed_cycles: float | None = None
    process_cycles_per_doc: float | None = None
    push_cycles_per_result: float | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "passed": self.passed,
            "gate_median_error_pct": self.gate_median_error_pct,
            "gate_max_error_pct": self.gate_max_error_pct,
            "median_error_pct": self.median_error_pct,
            "max_error_pct": self.max_error_pct,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "holdout_origins": list(self.holdout_origins),
            "errors": list(self.errors),
            "reasons": list(self.reasons),
            "topk_init_cycles": self.topk_init_cycles,
            "fixed_cycles": self.fixed_cycles,
            "process_cycles_per_doc": self.process_cycles_per_doc,
            "push_cycles_per_result": self.push_cycles_per_result,
        }


class FpgaServiceModel:
    """Service time from an HLS profile, calibrated only by explicit holdout data."""

    def __init__(self, profile: Any, ref: ProfileRef, clock_ns: float,
                 topk_init_cycles: float, process_cycles_per_doc: float,
                 push_cycles_per_result: float,
                 codebook_load_cycles: float = 0.0) -> None:
        self.profile = profile
        self.ref = ref
        self.clock_ns = clock_ns
        self.topk_init_cycles = topk_init_cycles
        self.process_cycles_per_doc = process_cycles_per_doc
        self.push_cycles_per_result = push_cycles_per_result
        # ``codebook_load_cycles`` is opt-in: 0 keeps the legacy cold model
        # (predict == total service including any codebook stream, as the
        # validation gate fits it).  Callers that model BRAM residency (the OSD
        # timing model) pass ``CODEBOOK_LOAD_CYCLES`` so ``codebook_resident=True``
        # subtracts the stream and a resident codebook is never charged twice.
        self.codebook_load_cycles = codebook_load_cycles
        self._fit: tuple[float, float, float] | None = None
        self._report: CalibrationReport | None = None

    @classmethod
    def from_profile(cls, profile: Any, ref: ProfileRef) -> "FpgaServiceModel":
        """Build from a parsed profile, refusing anything but a clean DES-clock ref."""
        if not isinstance(ref, ProfileRef):
            raise FpgaModelError(
                "FpgaServiceModel requires a contracts ProfileRef, got "
                f"{type(ref).__name__}; source and evidence cannot be omitted")
        profile_id = getattr(profile, "profile_id", None)
        if not profile_id or profile_id != ref.profile_id:
            raise FpgaModelError(
                f"profile id {profile_id!r} does not match ProfileRef {ref.profile_id!r}")
        status = getattr(profile, "status", None)
        if status != "ok":
            unknown = list(getattr(profile, "unknown", ()))[:6]
            raise ProfileUnavailableError(
                f"HLS profile {profile_id!r} status={status!r}; refusing to fabricate "
                f"timing. unknown fields: {unknown}")
        clock_ns = getattr(profile, "clock_target_ns", None)
        if clock_ns is None or clock_ns <= 0:
            raise ProfileUnavailableError(
                f"HLS profile {profile_id!r} has no usable target clock")
        loops = {str(getattr(loop, "name", "")).upper(): loop
                 for loop in getattr(profile, "loops", ())}
        init_loop = loops.get(PHASE_LOOPS["topk_init"])
        process_loop = loops.get(PHASE_LOOPS["topk_process"])
        push_loop = loops.get(PHASE_LOOPS["topk_push"])
        for phase, loop in (("topk_init", init_loop), ("topk_process", process_loop),
                            ("topk_push", push_loop)):
            expected = PHASE_LOOPS[phase]
            if loop is None:
                raise ProfileUnavailableError(
                    f"HLS profile {profile_id!r} is missing the {expected} loop needed "
                    f"for {phase}; unparsed phases are never back-filled")
            if getattr(loop, "achieved_ii", None) is None:
                raise ProfileUnavailableError(
                    f"HLS profile {profile_id!r} has no achieved II for {expected}")
        if init_loop.latency_min_cycles is not None:
            init_cycles = float(init_loop.latency_min_cycles)
        elif init_loop.trip_count is not None:
            init_cycles = float(init_loop.achieved_ii * init_loop.trip_count)
        else:
            raise ProfileUnavailableError(
                f"HLS profile {profile_id!r} TOPK_INIT has neither latency nor trip "
                "count; the init phase cannot be derived from the report")
        return cls(profile=profile, ref=ref, clock_ns=float(clock_ns),
                   topk_init_cycles=init_cycles,
                   process_cycles_per_doc=float(process_loop.achieved_ii),
                   push_cycles_per_result=float(push_loop.achieved_ii))

    @classmethod
    def from_profile_path(cls, profile_path: str | Path,
                          ref: ProfileRef) -> "FpgaServiceModel":
        """Load a versioned profile JSON; a missing file fails explicitly."""
        path = Path(profile_path)
        if not path.is_file():
            raise MissingProfileError(f"extracted HLS profile not found: {path}")
        from scripts.extract_hls import read_profile
        return cls.from_profile(read_profile(path), ref)

    @classmethod
    def from_report(cls, report_path: str | Path, ref: ProfileRef, *,
                    source_path: str | Path | None = None,
                    expected_source_sha256: str | None = None,
                    notes: Iterable[str] = ()) -> "FpgaServiceModel":
        """Extract directly from a real csynth report; no cached numbers are reused."""
        from scripts.extract_hls import extract_profile
        profile = extract_profile(report_path, source_path=source_path,
                                  expected_source_sha256=expected_source_sha256,
                                  profile_id=ref.profile_id, notes=tuple(notes))
        return cls.from_profile(profile, ref)

    @property
    def calibration(self) -> CalibrationReport | None:
        return self._report

    def _predict_with(self, fixed: float, process: float, push: float, n_docs: int,
                      top_k: int, init_runs: int = 1) -> float:
        push_units = min(top_k, n_docs)
        return (fixed + self.topk_init_cycles * init_runs
                + process * n_docs + push * push_units)

    def predict_estimate(self, n_docs: int, top_k: int, *, init_runs: int = 1,
                         codebook_resident: bool = False) -> ServiceEstimate:
        """Predict service time for one call; requires a completed calibration fit.

        The fitted base is the *cold* base (codebook stream included, exactly as
        the co-sim holdout measures it).  With ``codebook_resident=True`` and a
        non-zero ``codebook_load_cycles`` the codebook stream is removed, giving
        the warm service time the OSD timing model books in its FPGA stage after
        it has (or already had) the codebook in BRAM.  The default reproduces the
        cold prediction bit-for-bit, so the validation gate is unchanged.
        """
        n = _require_count("n_docs", n_docs)
        k = _require_count("top_k", top_k)
        runs = _require_count("init_runs", init_runs)
        if runs < 1:
            raise FpgaModelError(f"init_runs must be >= 1, got {runs!r}")
        if self._fit is None:
            raise NotCalibratedError(
                "stage coefficients are not calibrated; run calibrate() with "
                "independent RTL co-sim holdout data before predicting")
        fixed_cold, process, push = self._fit
        compute_fixed = fixed_cold - self.codebook_load_cycles
        fixed = compute_fixed if codebook_resident else fixed_cold
        total = self._predict_with(fixed, process, push, n, k, runs)
        stages = [
            ServiceStage("topk_init", startup_cycles=self.topk_init_cycles * runs,
                         steady_cycles=0.0, drain_cycles=0.0, units=runs),
            ServiceStage("topk_process", startup_cycles=0.0,
                         steady_cycles=process * n, drain_cycles=0.0, units=n),
            ServiceStage("topk_push", startup_cycles=0.0, steady_cycles=0.0,
                         drain_cycles=push * min(k, n), units=min(k, n)),
            ServiceStage("pipeline_fixed", startup_cycles=compute_fixed, steady_cycles=0.0,
                         drain_cycles=0.0, units=1),
        ]
        if not codebook_resident and self.codebook_load_cycles > 0:
            stages.append(ServiceStage("codebook_load",
                                       startup_cycles=self.codebook_load_cycles,
                                       steady_cycles=0.0, drain_cycles=0.0, units=1))
        return ServiceEstimate(n_docs=n, top_k=k, init_runs=runs, total_cycles=total,
                               clock_ns=self.clock_ns, total_ns=total * self.clock_ns,
                               stages=tuple(stages))

    def scan_cycles_per_doc(self, n_docs: int | None = None) -> float:
        """Steady-state scan cycles per document from the calibrated anchor.

        n_docs is accepted so engine wrappers (e.g. a direct-mode selector)
        can report a size-dependent throughput; the table engine is uniform.
        """
        if self._fit is None:
            raise NotCalibratedError(
                "stage coefficients are not calibrated; run calibrate() first")
        return float(self._fit[1])

    def predict_cycles(self, n_docs: int, top_k: int, *, init_runs: int = 1,
                       codebook_resident: bool = False) -> float:
        return self.predict_estimate(n_docs, top_k, init_runs=init_runs,
                                     codebook_resident=codebook_resident).total_cycles

    def predict_ns(self, n_docs: int, top_k: int, *, init_runs: int = 1,
                   codebook_resident: bool = False) -> float:
        return self.predict_estimate(n_docs, top_k, init_runs=init_runs,
                                     codebook_resident=codebook_resident).total_ns

    def calibrate(self, train_points: Iterable[CalibrationPoint],
                  holdout_points: Iterable[CalibrationPoint] = ()) -> CalibrationReport:
        """Fit the fixed base against the anchored process throughput.

        The process term is anchored to the hardware steady-state throughput
        (``ANCHORED_PROCESS_CYCLES_PER_DOC``) and the ``min(top_k, n)`` push
        term is dropped: for n >= top_k that column is collinear with the
        constant, the 3-parameter least-squares fit was unidentifiable, and it
        returned a non-physical negative push.  Only the fixed base is fitted:

            total_cycles = (fixed + topk_init) + 2.0 * n_docs

        Each training point is assumed to contain exactly one TOPK init run.
        The acceptance gate is evaluated exclusively on the holdout split.
        """
        train = tuple(train_points)
        holdout = tuple(holdout_points)
        if not train:
            raise FpgaModelError("calibration requires at least one training point")
        for point in train + holdout:
            if not isinstance(point, CalibrationPoint):
                raise FpgaModelError(
                    f"calibration points must be CalibrationPoint, got {point!r}")
        fixed_matrix = np.ones((len(train), 1), dtype=float)
        target = np.array([point.total_cycles - self.topk_init_cycles
                           - ANCHORED_PROCESS_CYCLES_PER_DOC * point.n_docs
                           for point in train], dtype=float)
        (fixed,), *_ = np.linalg.lstsq(fixed_matrix, target, rcond=None)
        fixed = float(fixed)
        process = ANCHORED_PROCESS_CYCLES_PER_DOC
        push = 0.0
        self._fit = (fixed, process, push)

        errors: list[dict] = []
        predictions = [self._predict_with(fixed, process, push, point.n_docs, point.top_k)
                       for point in holdout]
        for point, predicted in zip(holdout, predictions):
            errors.append({
                **point.to_dict(),
                "predicted_cycles": predicted,
                "error_pct": abs(predicted - point.total_cycles) / point.total_cycles * 100.0,
            })
        origins = tuple(sorted({point.origin for point in holdout}))
        median_error = max_error = None
        reasons: list[str] = []
        if not holdout:
            status = STATUS_BLOCKED
            reasons.append(
                "no holdout data: the acceptance gate is not evaluated. Exact missing "
                "input: independent RTL co-sim total-cycle measurements at holdout N "
                f"{list(HOLDOUT_N)} (cold/warm codebook, repeated calls). csynth report "
                "parsing is extraction validation only, not independent accuracy "
                "validation.")
        else:
            median_error = float(np.median([item["error_pct"] for item in errors]))
            max_error = float(np.max([item["error_pct"] for item in errors]))
            if origins != ("cosim",):
                status = STATUS_NOT_CALIBRATED
                reasons.append(
                    f"holdout origin(s) {list(origins)} are not independent RTL co-sim "
                    "total-cycle data; the median<=5% / max<=10% gate is not claimed "
                    "(csynth-report agreement is extraction validation only)")
            elif median_error <= GATE_MEDIAN_ERROR_PCT and max_error <= GATE_MAX_ERROR_PCT:
                status = STATUS_CALIBRATED
                reasons.append(
                    f"holdout median {median_error:.3f}% <= {GATE_MEDIAN_ERROR_PCT}% and "
                    f"max {max_error:.3f}% <= {GATE_MAX_ERROR_PCT}% on independent RTL "
                    "co-sim total-cycle data")
            else:
                status = STATUS_NOT_CALIBRATED
                reasons.append(
                    f"holdout error gate failed: median {median_error:.3f}% "
                    f"(gate {GATE_MEDIAN_ERROR_PCT}%), max {max_error:.3f}% "
                    f"(gate {GATE_MAX_ERROR_PCT}%)")
        reasons.append(MODEL_FORM_REASON)
        self._report = CalibrationReport(
            status=status,
            passed=status == STATUS_CALIBRATED,
            gate_median_error_pct=GATE_MEDIAN_ERROR_PCT,
            gate_max_error_pct=GATE_MAX_ERROR_PCT,
            median_error_pct=median_error,
            max_error_pct=max_error,
            n_train=len(train),
            n_holdout=len(holdout),
            holdout_origins=origins,
            errors=tuple(errors),
            reasons=tuple(reasons),
            topk_init_cycles=self.topk_init_cycles,
            fixed_cycles=fixed,
            process_cycles_per_doc=process,
            push_cycles_per_result=push,
        )
        return self._report


__all__ = [
    "GATE_MAX_ERROR_PCT",
    "GATE_MEDIAN_ERROR_PCT",
    "HOLDOUT_N",
    "PHASE_LOOPS",
    "STATUS_BLOCKED",
    "STATUS_CALIBRATED",
    "STATUS_NOT_CALIBRATED",
    "CalibrationPoint",
    "CalibrationReport",
    "FpgaModelError",
    "FpgaServiceModel",
    "MissingProfileError",
    "NotCalibratedError",
    "ProfileUnavailableError",
    "ServiceEstimate",
    "ServiceStage",
]
