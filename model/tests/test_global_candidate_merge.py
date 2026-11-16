"""Synthetic contract/oracle tests; no actual FiQA per-list evidence."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import itertools
import random
import unittest

from scripts.global_candidate_merge import (
    Candidate, GlobalCandidateMerger, ListResult, ListSpec, MergeError, QueryIdentity,
)


def identity(r=3, n=2):
    return QueryIdentity(index_namespace="synthetic", index_version="v1",
        index_sha256="1" * 64, query_id="q", query_sha256="2" * 64,
        document_vectors_sha256="3" * 64, ownership_sha256="4" * 64,
        profile_sha256="5" * 64, encoder_revision="6" * 40,
        metric="IP", top_r=r, top_k=1, nprobe=n, by_residual=True,
        search_config=(("adc_revision", "synthetic-v1"),))


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.ident = identity()
        self.specs = (ListSpec("a", 0, 10, 4), ListSpec("b", 10, 20, 4))
        self.rows = ((Candidate(1, 9.), Candidate(2, 7.), Candidate(3, 5.)),
                     (Candidate(11, 8.), Candidate(12, 7.), Candidate(13, 6.)))

    def merger(self, **kwargs):
        return GlobalCandidateMerger(self.ident, self.specs, **kwargs)

    def packet(self, i, **kwargs):
        spec = self.specs[i]
        return replace(ListResult(self.ident, spec.list_id, spec.cardinality,
                                  len(self.rows[i]), self.rows[i]), **kwargs)

    def assert_poisoned(self, merger):
        self.assertEqual(merger.state, "failed")
        with self.assertRaises(MergeError):
            merger.finish()
        with self.assertRaises(MergeError):
            merger.consume(self.packet(0))

    def test_shuffled_arrival_and_no_early_output(self):
        for order in ((0, 1), (1, 0)):
            m = self.merger()
            for i in order:
                self.assertIsNone(m.consume(self.packet(i)))
                self.assertEqual(m.state, "accepting")
            self.assertEqual(m.finish(), (Candidate(1, 9.), Candidate(11, 8.), Candidate(2, 7.)))
            self.assertEqual(m.finish(), m.finish())

    def test_ties_at_local_and_global_boundary(self):
        m = self.merger()
        m.consume(self.packet(1, candidates=tuple(Candidate(i, 7.) for i in (10, 11, 12))))
        m.consume(self.packet(0, candidates=tuple(Candidate(i, 7.) for i in (0, 1, 2))))
        self.assertEqual(tuple(c.row_id for c in m.finish()), (0, 1, 2))

    def test_short_and_empty_lists(self):
        specs = (ListSpec("a", 0, 10, 1), ListSpec("b", 10, 20, 0))
        m = GlobalCandidateMerger(self.ident, specs)
        m.consume(ListResult(self.ident, "b", 0, 0, ()))
        m.consume(ListResult(self.ident, "a", 1, 1, (Candidate(0, -1.),)))
        self.assertEqual(m.finish(), (Candidate(0, -1.),))

    def test_missing_list_closes_fail_closed(self):
        m = self.merger()
        m.consume(self.packet(0))
        with self.assertRaisesRegex(MergeError, "missing"):
            m.finish()
        self.assert_poisoned(m)

    def test_error_discards_entire_query(self):
        m = self.merger()
        m.consume(self.packet(0))
        with self.assertRaises(MergeError):
            m.consume(self.packet(1, error="backend failed"))
        self.assert_poisoned(m)

    def test_all_empty_lists(self):
        specs = tuple(replace(s, cardinality=0) for s in self.specs)
        m = GlobalCandidateMerger(self.ident, specs)
        for s in reversed(specs):
            m.consume(ListResult(self.ident, s.list_id, 0, 0, ()))
        self.assertEqual(m.finish(), ())

    def test_valid_but_different_hashes_and_search_config(self):
        for field in ("index_sha256", "query_sha256", "document_vectors_sha256",
                      "ownership_sha256", "profile_sha256", "encoder_revision"):
            m = self.merger()
            wrong = replace(self.ident, **{field: "a" * len(getattr(self.ident, field))})
            with self.assertRaisesRegex(MergeError, "identity mismatch"):
                m.consume(self.packet(0, identity=wrong))
            self.assert_poisoned(m)

    def test_malformed_packets_fail_closed(self):
        for packet in (None, self.packet(0, candidates=list(self.rows[0])),
                       self.packet(0, candidates=(None,) + self.rows[0][1:]),
                       self.packet(0, error=False)):
            m = self.merger()
            with self.assertRaises(MergeError):
                m.consume(packet)
            self.assert_poisoned(m)

    def test_duplicate_and_unexpected_lists(self):
        for packet in (self.packet(0), self.packet(1, list_id="unexpected")):
            m = self.merger()
            m.consume(self.packet(0))
            with self.assertRaises(MergeError):
                m.consume(packet)
            self.assert_poisoned(m)

    def test_crosslist_duplicate_rejected_by_ownership(self):
        m = self.merger()
        m.consume(self.packet(0))
        with self.assertRaisesRegex(MergeError, "ownership"):
            m.consume(self.packet(1, candidates=(Candidate(1, 9.), Candidate(12, 7.), Candidate(13, 6.))))
        self.assert_poisoned(m)

    def test_every_identity_field_matches(self):
        for field in self.ident.__dataclass_fields__:
            old = getattr(self.ident, field)
            value = (not old if type(old) is bool else old + 1 if type(old) is int
                     else (("adc_revision", "different"),) if type(old) is tuple
                     else "different")
            with self.subTest(field=field):
                m = self.merger()
                with self.assertRaises(MergeError):
                    m.consume(self.packet(0, identity=replace(self.ident, **{field: value})))
                self.assertEqual(m.consumed_count, 0)
                self.assert_poisoned(m)

    def test_invalid_scores_and_uint64_rows(self):
        for candidate in (Candidate(0, float("nan")), Candidate(0, float("inf")),
                          Candidate(0, True), Candidate(True, 10.), Candidate(-1, 10.),
                          Candidate(2**64, 10.), Candidate(0.0, 10.),
                          Candidate(0, 10**1000)):
            with self.subTest(candidate=candidate):
                m = self.merger()
                with self.assertRaises(MergeError):
                    m.consume(self.packet(0, candidates=(candidate,) + self.rows[0][1:]))
                self.assertEqual(m.consumed_count, 0)
                self.assert_poisoned(m)

    def test_uint64_max_and_finite_integer_score(self):
        ident = identity(1, 1)
        m = GlobalCandidateMerger(ident, (ListSpec("last", 2**64-1, 2**64, 1),))
        row = Candidate(2**64-1, 2)
        m.consume(ListResult(ident, "last", 1, 1, (row,)))
        self.assertEqual(m.finish(), (row,))

    def test_bad_order_and_local_duplicates(self):
        for rows in ((Candidate(2, 7.), Candidate(1, 7.), Candidate(3, 5.)),
                     tuple(reversed(self.rows[0])),
                     (Candidate(1, 9.), Candidate(1, 7.), Candidate(3, 5.))):
            m = self.merger()
            with self.assertRaises(MergeError):
                m.consume(self.packet(0, candidates=rows))
            self.assert_poisoned(m)

    def test_insufficient_depth_and_bad_counts(self):
        for changes in ({"local_depth": 2, "candidates": self.rows[0][:2]},
                        {"local_depth": 4}, {"cardinality": 3},
                        {"cardinality": True}, {"local_depth": True},
                        {"local_depth": 2**64}):
            m = self.merger()
            with self.assertRaises(MergeError):
                m.consume(self.packet(0, **changes))
            self.assertEqual(m.consumed_count, 0)
            self.assert_poisoned(m)

    def test_limits_checked_before_barrier_mutation(self):
        with self.assertRaises(MergeError):
            self.merger(max_lists=1)
        with self.assertRaises(MergeError):
            self.merger(max_candidates=5)
        m = self.merger(max_candidates=6)
        m.consume(self.packet(0))
        with self.assertRaises(MergeError):
            m.consume(self.packet(1, local_depth=4,
                candidates=self.rows[1] + (Candidate(14, 0.),)))
        self.assertEqual(m.consumed_count, 1)
        self.assert_poisoned(m)

    def test_immutable_configuration_and_owned_results(self):
        with self.assertRaises(FrozenInstanceError):
            self.ident.top_r = 1
        with self.assertRaises(FrozenInstanceError):
            self.specs[0].cardinality = 1
        m = self.merger()
        for i in range(2):
            m.consume(self.packet(i))
        with self.assertRaises(FrozenInstanceError):
            m.finish()[0].row_id = 99

    def test_invalid_manifests(self):
        for specs in ((self.specs[0], self.specs[0]),
                      (self.specs[0], ListSpec("b", 9, 20, 4)),
                      (self.specs[0], ListSpec("b", 10, 20, 11)),
                      (self.specs[0], ListSpec("b", True, 20, 4)),
                      list(self.specs)):
            with self.assertRaises(MergeError):
                GlobalCandidateMerger(self.ident, specs)
        for changes in ({"metric": "L2"}, {"query_sha256": "bad"},
                        {"document_vectors_sha256": "bad"}, {"ownership_sha256": "bad"},
                        {"top_r": True}, {"nprobe": 1}, {"search_config": []},
                        {"search_config": (("x", "a"), ("x", "b"))}):
            with self.assertRaises(MergeError):
                GlobalCandidateMerger(replace(self.ident, **changes), self.specs)
        for kwargs in ({"max_lists": True}, {"max_candidates": 2**64}):
            with self.assertRaises(MergeError):
                self.merger(**kwargs)

    def test_aggregate_cardinality_overflow(self):
        ident = identity(1, 2)
        # Overlapping ownership is independently invalid; neither may be accepted.
        with self.assertRaises(MergeError):
            GlobalCandidateMerger(ident, (ListSpec("a", 0, 2**64, 2**64-1),
                                         ListSpec("b", 0, 2, 2)))

    def test_more_than_local_r_is_allowed(self):
        m = self.merger()
        m.consume(self.packet(0, local_depth=4,
            candidates=self.rows[0] + (Candidate(4, 0.),)))
        m.consume(self.packet(1))
        self.assertEqual(tuple(c.row_id for c in m.finish()), (1, 11, 2))

    def test_synthetic_r64_uses_global_not_per_list_output(self):
        ident = identity(64, 2)
        specs = (ListSpec("a", 0, 100, 100), ListSpec("b", 100, 200, 100))
        m = GlobalCandidateMerger(ident, specs)
        for s in reversed(specs):
            rows = tuple(Candidate(i, 1.) for i in range(s.row_start, s.row_start + 64))
            m.consume(ListResult(ident, s.list_id, 100, 64, rows))
        self.assertEqual(tuple(c.row_id for c in m.finish()), tuple(range(64)))

    def test_late_input_invalidates_closed_owner(self):
        m = self.merger()
        for i in range(2):
            m.consume(self.packet(i))
        m.finish()
        with self.assertRaises(MergeError):
            m.consume(self.packet(0))
        self.assert_poisoned(m)

    def test_property_full_union_across_seeds_and_all_arrival_orders(self):
        for seed in range(30):
            rng = random.Random(seed)
            r = rng.randint(1, 12)
            ident = identity(r, 4)
            full = [tuple(sorted((Candidate(i*100+j, rng.randint(-3, 3))
                    for j in range(rng.randint(0, 25))), key=lambda c: (-c.score, c.row_id)))
                    for i in range(4)]
            specs = tuple(ListSpec(str(i), i*100, (i+1)*100, len(rows))
                          for i, rows in enumerate(full))
            expected = tuple(sorted(itertools.chain.from_iterable(full),
                                    key=lambda c: (-c.score, c.row_id))[:r])
            for order in itertools.permutations(range(4)):
                m = GlobalCandidateMerger(ident, specs)
                for i in order:
                    rows = full[i][:r]
                    m.consume(ListResult(ident, str(i), len(full[i]), len(rows), rows))
                self.assertEqual(m.finish(), expected, (seed, order))


class MembershipTests(unittest.TestCase):
    def setUp(self):
        # Keep the existing range regressions runnable during tests-first failure.
        from scripts import global_candidate_merge as module
        self.module = module
        self.specs = tuple(module.MembershipSpec(name, 3, rows) for name, rows in
                           (("a", [0, 2, 4]), ("b", [1, 3, 5])))

    def digest(self, specs):
        # Independent canonical binary oracle, not the production digest helper.
        data = b"cortex.membership-ownership.v1\0" + len(specs).to_bytes(8, "big")
        for s in specs:
            name = s.list_id.encode("utf-8")
            data += len(name).to_bytes(8, "big") + name
            data += s.cardinality.to_bytes(8, "big")
            data += b"".join(r.to_bytes(8, "big") for r in sorted(s.rows))
        return hashlib.sha256(data).hexdigest(), len(data)

    def make(self, specs=None, **kwargs):
        specs = self.specs if specs is None else specs
        ident = replace(identity(), ownership_sha256=self.digest(specs)[0])
        return GlobalCandidateMerger(ident, specs, **kwargs)

    def packet(self, m, name, rows, score=1.):
        return ListResult(m.identity, name, len(rows), len(rows),
                          tuple(Candidate(r, score) for r in rows))

    def test_interleaved_original_ties_shuffled_and_copied(self):
        for order in (("a", "b"), ("b", "a")):
            m = self.make()
            for name in order:
                rows = (0, 2, 4) if name == "a" else (1, 3, 5)
                self.assertIsNone(m.consume(self.packet(m, name, rows)))
            self.assertEqual(tuple(c.row_id for c in m.finish()), (0, 1, 2))
        m = self.make()
        self.specs[0].rows[:] = [99]
        self.assertEqual(m.required_lists[0].rows, (0, 2, 4))
        with self.assertRaises(FrozenInstanceError):
            m.required_lists[0].cardinality = 9
        m.consume(self.packet(m, "a", (0, 2, 4)))
        m.consume(self.packet(m, "b", (1, 3, 5)))
        self.assertEqual(tuple(c.row_id for c in m.finish()), (0, 1, 2))

    def test_canonical_hash_order_content_and_exact_byte_budget(self):
        digest, size = self.digest(self.specs)
        self.assertEqual(self.module.membership_ownership_sha256(self.specs), digest)
        unordered = (replace(self.specs[0], rows=[4, 0, 2]), self.specs[1])
        self.assertEqual(self.module.membership_ownership_sha256(unordered), digest)
        self.make(max_ownership_entries=6, max_ownership_bytes=size)
        for kwargs in ({"max_ownership_entries": 5}, {"max_ownership_bytes": size-1},
                       {"max_ownership_entries": True}, {"max_ownership_bytes": 2**64}):
            with self.assertRaises(MergeError):
                self.make(**kwargs)
        for specs in (tuple(reversed(self.specs)),
                      (replace(self.specs[0], rows=[0, 2, 6]), self.specs[1]),
                      (replace(self.specs[0], list_id="different"), self.specs[1])):
            with self.assertRaisesRegex(MergeError, "ownership digest"):
                GlobalCandidateMerger(replace(identity(), ownership_sha256=digest), specs)
        with self.assertRaisesRegex(MergeError, "ownership digest"):
            GlobalCandidateMerger(identity(), self.specs)

    def test_invalid_ownership_manifests(self):
        bad = [self.specs[:1], self.specs + (self.specs[0],),
               (self.specs[0], replace(self.specs[1], list_id="a")),
               (self.specs[0], replace(self.specs[1], rows=[1, 2, 5])),
               (replace(self.specs[0], rows=[0, 0, 4]), self.specs[1]),
               (replace(self.specs[0], cardinality=2), self.specs[1]),
               (replace(self.specs[0], cardinality=True), self.specs[1]),
               (replace(self.specs[0], rows=iter([0, 2, 4])), self.specs[1]),
               (self.specs[0], ListSpec("b", 10, 20, 3))]
        bad += [(replace(self.specs[0], rows=[0, 2, row]), self.specs[1])
                for row in (True, -1, 2**64, 4.0)]
        for specs in bad:
            with self.subTest(specs=specs), self.assertRaises(MergeError):
                GlobalCandidateMerger(identity(), specs)

    def test_wrong_owner_unknown_row_and_dropped_duplicate_fail_closed(self):
        for rows in ((0, 3, 5), (1, 3, 99), (1, 3, 4)):
            m = self.make()
            # Supply three a rows to an R1 heap: row4 is already dropped before b.
            ident = replace(m.identity, top_r=1)
            m = GlobalCandidateMerger(ident, self.specs)
            m.consume(self.packet(m, "a", (0, 2, 4), score=-1.))
            with self.assertRaisesRegex(MergeError, "ownership"):
                m.consume(self.packet(m, "b", rows))
            self.assertEqual(m.consumed_count, 1)
            self.assertEqual(m.state, "failed")
            with self.assertRaises(MergeError):
                m.finish()

    def test_independent_ownership_budget_and_all_empty(self):
        # Ownership is six rows even when the candidate budget permits only two.
        ident = replace(identity(1), ownership_sha256=self.digest(self.specs)[0])
        m = GlobalCandidateMerger(ident, self.specs, max_candidates=2,
                                  max_ownership_entries=6)
        m.consume(ListResult(ident, "a", 3, 1, (Candidate(4, 2.),)))
        m.consume(ListResult(ident, "b", 3, 1, (Candidate(1, 2.),)))
        self.assertEqual(m.finish(), (Candidate(1, 2.),))
        with self.assertRaisesRegex(MergeError, "ownership entry budget"):
            GlobalCandidateMerger(ident, self.specs, max_candidates=100,
                                  max_ownership_entries=5)
        specs = tuple(replace(s, cardinality=0, rows=[]) for s in self.specs)
        m = self.make(specs, max_ownership_entries=0)
        for name in ("b", "a"):
            m.consume(self.packet(m, name, ()))
        self.assertEqual(m.finish(), ())

    def test_packet_ownership_hash_must_match_trusted_identity(self):
        m = self.make()
        packet = self.packet(m, "a", (0, 2, 4))
        with self.assertRaisesRegex(MergeError, "identity mismatch"):
            m.consume(replace(packet, identity=replace(m.identity, ownership_sha256="a"*64)))
        self.assertEqual(m.consumed_count, 0)
        self.assertEqual(m.state, "failed")

    def test_membership_barrier_and_unexpected_list(self):
        for action in ("missing", "unexpected", "error", "duplicate"):
            m = self.make()
            p = self.packet(m, "a", (0, 2, 4))
            m.consume(p)
            with self.assertRaises(MergeError):
                if action == "missing":
                    m.finish()
                else:
                    m.consume(p if action == "duplicate" else
                              replace(p, list_id="extra" if action == "unexpected" else "b",
                                      error="failure" if action == "error" else None))
            self.assertEqual(m.state, "failed")

    def test_empty_membership_and_uint64_max(self):
        specs = (self.module.MembershipSpec("a", 0, []),
                 self.module.MembershipSpec("b", 1, [2**64-1]))
        m = self.make(specs)
        m.consume(self.packet(m, "a", ()))
        m.consume(self.packet(m, "b", (2**64-1,)))
        self.assertEqual(m.finish(), (Candidate(2**64-1, 1.),))

    def test_property_interleaved_full_union_all_arrivals(self):
        for seed in range(20):
            rng = random.Random(seed)
            owned = [[] for _ in range(4)]
            for row in rng.sample(range(200), 70):
                owned[rng.randrange(4)].append(row)
            specs = tuple(self.module.MembershipSpec(str(i), len(rows), rows)
                          for i, rows in enumerate(owned))
            ident = replace(identity(rng.randint(1, 10), 4),
                            ownership_sha256=self.digest(specs)[0])
            full = [tuple(sorted((Candidate(r, rng.randint(-3, 3)) for r in rows),
                                 key=lambda c: (-c.score, c.row_id))) for rows in owned]
            expected = tuple(sorted(itertools.chain.from_iterable(full),
                                    key=lambda c: (-c.score, c.row_id))[:ident.top_r])
            for order in itertools.permutations(range(4)):
                m = GlobalCandidateMerger(ident, specs)
                for i in order:
                    local = full[i][:ident.top_r]
                    m.consume(ListResult(ident, str(i), len(full[i]), len(local), local))
                self.assertEqual(m.finish(), expected, (seed, order))


if __name__ == "__main__":
    unittest.main()
