"""Bounded-wait cache benefit uses live savings, not job size."""
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from strata_sim.sched.query_queue import QueryQueue, QueuedQuery, OsdView


class BoundedCacheTest(unittest.TestCase):
    def queue(self):
        q = QueryQueue(strategy='bwc', max_wait_ns=100)
        q.push(QueuedQuery('old', 'a', 1, arrival_ns=0, estimated_compute_ns=1))
        q.push(QueuedQuery('hit', 'b', 2, arrival_ns=10, estimated_compute_ns=10000))
        return q

    def test_benefit_over_short_job(self):
        v = OsdView(now_ns=20, cache_savings_ns={'old': 0, 'hit': 500})
        self.assertEqual(self.queue().pop(20, v).query_id, 'hit')

    def test_hard_wait_override(self):
        v = OsdView(now_ns=100, cache_savings_ns={'old': 0, 'hit': 500})
        self.assertEqual(self.queue().pop(100, v).query_id, 'old')

    def test_changed_residency_changes_selection(self):
        v = OsdView(now_ns=20, cache_savings_ns={'old': 700, 'hit': 0})
        self.assertEqual(self.queue().pop(20, v).query_id, 'old')

    def test_peek_preserves_state(self):
        q = self.queue()
        v = OsdView(now_ns=20, cache_savings_ns={'old': 0, 'hit': 500})
        self.assertEqual(q.peek(20, v).query_id, q.pop(20, v).query_id)
        self.assertEqual(q.pending, 1)

    def test_cbw_prefers_cache_benefit_per_service(self):
        q = QueryQueue(strategy='cbw', max_wait_ns=100)
        q.push(QueuedQuery('large-hit', 'a', 1, arrival_ns=10))
        q.push(QueuedQuery('small-hit', 'b', 2, arrival_ns=11))
        v = OsdView(
            now_ns=20,
            service_estimates_ns={'large-hit': 1000, 'small-hit': 100},
            cache_savings_ns={'large-hit': 500, 'small-hit': 100},
        )
        self.assertEqual(q.pop(20, v).query_id, 'small-hit')

    def test_cbw_requires_live_service_and_savings(self):
        q = QueryQueue(strategy='cbw')
        q.push(QueuedQuery('q', 'a', 1))
        with self.assertRaisesRegex(Exception, 'cbw requires live'):
            q.pop(0, OsdView(now_ns=0, cache_savings_ns={'q': 1}))

    def test_cbw_order_matches_viewed_pop(self):
        q = QueryQueue(strategy='cbw', max_wait_ns=100)
        q.push(QueuedQuery('a', 'a', 1, arrival_ns=1))
        q.push(QueuedQuery('b', 'b', 2, arrival_ns=2))
        v = OsdView(
            now_ns=3,
            service_estimates_ns={'a': 100, 'b': 1000},
            cache_savings_ns={'a': 50, 'b': 900},
        )
        self.assertEqual(q.order(3, v), ('b', 'a'))

    def test_cbw_order_preserves_overdue_priority(self):
        q = QueryQueue(strategy='cbw', max_wait_ns=100)
        q.push(QueuedQuery('old', 'a', 1, arrival_ns=0))
        q.push(QueuedQuery('hit', 'b', 2, arrival_ns=90))
        v = OsdView(
            now_ns=100,
            service_estimates_ns={'old': 1000, 'hit': 1},
            cache_savings_ns={'old': 0, 'hit': 1000},
        )
        self.assertEqual(q.order(100, v), ('old', 'hit'))

    def test_cbw_peek_preserves_state(self):
        q = QueryQueue(strategy='cbw', max_wait_ns=100)
        q.push(QueuedQuery('a', 'a', 1, arrival_ns=1))
        q.push(QueuedQuery('b', 'b', 2, arrival_ns=2))
        v = OsdView(
            now_ns=3,
            service_estimates_ns={'a': 100, 'b': 1000},
            cache_savings_ns={'a': 50, 'b': 900},
        )
        self.assertEqual(q.peek(3, v).query_id, 'b')
        self.assertEqual(q.pending, 2)

    def test_negative_max_wait_is_rejected(self):
        with self.assertRaisesRegex(Exception, 'max_wait_ns'):
            QueryQueue(strategy='cbw', max_wait_ns=-1)


if __name__ == '__main__':
    unittest.main()
