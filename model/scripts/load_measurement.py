"""Independent open-loop arrivals and explicit half-open measurement windows."""
import math
from dataclasses import replace
import numpy as np


def uniform_arrivals(queries, qps):
    if not math.isfinite(qps) or qps <= 0:
        raise ValueError('qps must be finite and positive')
    return [replace(q, arrival_ns=round(i * 1e9 / qps),
                    deadline_ns=round(i * 1e9 / qps) + q.deadline_ns-q.arrival_ns)
            for i, q in enumerate(queries)]


def window_metrics(completed, start_ns, end_ns):
    if end_ns <= start_ns:
        raise ValueError('empty measurement interval')
    arrivals = [x for x in completed if start_ns <= x.query.arrival_ns < end_ns]
    finishes = [x for x in completed if start_ns <= x.finish_ns < end_ns]
    def backlog(t):
        return sum(x.query.arrival_ns < t and x.finish_ns >= t for x in completed)
    lat = [x.finish_ns-x.query.arrival_ns for x in arrivals]
    b0, b1 = backlog(start_ns), backlog(end_ns)
    return dict(start_ns=start_ns, end_ns=end_ns,
                completion_qps=len(finishes)*1e9/(end_ns-start_ns),
                arrival_count=len(arrivals), completion_count=len(finishes),
                outstanding_start=b0, outstanding_end=b1,
                outstanding_growth=b1-b0,
                arrival_cohort_p50_ms=float(np.percentile(lat, 50)/1e6) if lat else None,
                arrival_cohort_p99_ms=float(np.percentile(lat, 99)/1e6) if lat else None,
                arrival_cohort_followed_through_drain=True,
                diagnosis='growing_backlog' if b1>b0 else 'finite_window_no_backlog_growth',
                steady_state_proven=False)
