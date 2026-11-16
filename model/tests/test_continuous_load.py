import pathlib
import sys
import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from load_measurement import uniform_arrivals, window_metrics
from test_profile_consistency import FakeFpga
from strata_sim.des.engine import DesEngine
from strata_sim.sched.osd_timing import OSDHardwareProfile, OSDTimingModel, add_osd_resources
from strata_sim.sched.query_queue import QueryQueue, QueuedQuery, OsdView


@dataclass
class Q:
    arrival_ns: int
    deadline_ns: int


class ContinuousLoadTest(unittest.TestCase):
    def test_uniform_ignores_trace_gaps_preserves_budgets(self):
        qs = uniform_arrivals([Q(10, 60), Q(10000, 10070), Q(999999, 1000099)], 1000)
        self.assertEqual([q.arrival_ns for q in qs], [0, 1000000, 2000000])
        self.assertEqual([q.deadline_ns-q.arrival_ns for q in qs], [50, 70, 100])

    def test_invalid_rate(self):
        for rate in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                uniform_arrivals([], rate)

    def test_window_boundaries_and_conservation(self):
        c = [SimpleNamespace(query=Q(a, a+10), finish_ns=f)
             for a, f in [(0, 10), (10, 20), (19, 30), (20, 40)]]
        m = window_metrics(c, 10, 20)
        self.assertEqual(m['completion_count'], 1)
        self.assertEqual(m['arrival_count'], 2)
        self.assertEqual(m['outstanding_start'], 1)
        self.assertEqual(m['outstanding_end'], 2)
        self.assertEqual(m['completion_qps'], 1e8)
        self.assertEqual(m['outstanding_growth'], m['arrival_count']-m['completion_count'])
        self.assertFalse(m['steady_state_proven'])
        self.assertAlmostEqual(m['arrival_cohort_p50_ms'], 10.5/1e6)

    def test_empty_cohort_and_finish_at_right_boundary(self):
        c = [SimpleNamespace(query=Q(0, 100), finish_ns=20)]
        m = window_metrics(c, 10, 20)
        self.assertEqual(m['completion_count'], 0)
        self.assertEqual(m['outstanding_end'], 1)
        self.assertIsNone(m['arrival_cohort_p99_ms'])

    def test_eviction_order_changes_real_transfer_count(self):
        def execute(order):
            engine = DesEngine()
            add_osd_resources(engine, 0)
            hw = replace(OSDHardwareProfile.from_model('ideal'),
                         pq_cache_bytes=128064)
            tm = OSDTimingModel(0, engine, FakeFpga(), hw,
                               cluster_sizes={0: 1000, 1: 1000},
                               ddr_cached_clusters=[0], bram_codebooks=['cb'])
            end = 0
            count = 0
            for i, cid in enumerate(order):
                t = tm.simulate_query(str(i), [cid], 10, codebook_id='cb', start_ns=end)
                end = t.end_ns
                count += t.ddr_transfers
            return count, tm.ddr_resident()
        self.assertEqual(execute([0, 1])[0], 1)
        self.assertEqual(execute([1, 0])[0], 2)
        self.assertEqual(execute([1, 0])[1], frozenset({0}))

    def test_bram_eviction_counts_only_replacement(self):
        engine = DesEngine()
        add_osd_resources(engine, 0)
        hw = replace(OSDHardwareProfile.from_model('ideal'), bram_capacity_slots=1)
        tm = OSDTimingModel(0, engine, FakeFpga(), hw,
                            cluster_sizes={0: 1000},
                            ddr_cached_clusters=[0], bram_codebooks=[])

        first = tm.simulate_query('first', [0], 10, codebook_id='cb0', start_ns=0)
        second = tm.simulate_query('second', [0], 10, codebook_id='cb1',
                                   start_ns=first.end_ns)
        hit = tm.simulate_query('hit', [0], 10, codebook_id='cb1',
                                start_ns=second.end_ns)

        self.assertEqual(first.bram_evictions, 0)
        self.assertEqual(second.bram_evictions, 1)
        self.assertEqual(hit.bram_evictions, 0)

    def test_batch_reselects_earliest_group_not_persistent_batch(self):
        q = QueryQueue(strategy='batch')
        for name, cb, at in [('a', 'A', 0), ('b', 'B', 1), ('c', 'A', 2)]:
            q.push(QueuedQuery(name, 'tenant', 0, arrival_ns=at, codebook_id=cb))
        self.assertEqual([q.pop(10).query_id for _ in range(3)], ['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main()
