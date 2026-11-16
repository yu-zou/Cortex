"""Synthetic contract tests do not provide workload cost/quality evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.query_cost_alignment import assess_alignment


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        def artifact(name, value):
            p = self.root / name
            p.write_text(json.dumps(value))
            return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        self.artifact = artifact
        self.expected = dict(index_namespace="fixture", index_version="v1",
            index_sha256="1" * 64, query_sha256="2" * 64,
            profile_sha256="3" * 64, query_id="q", nprobe=2, top_r=64, top_k=10,
            metric="IP", by_residual=True, encoder_revision="4" * 40)
        self.quality = dict(identity=copy.deepcopy(self.expected),
            candidate_scope="query_global_unique", query_count=1,
            rows=artifact("rows.json", [dict(query_id="q", query_sha256="2" * 64,
                candidate_ids=list(range(64)), candidate_scores=list(range(64)),
                result_ids=list(range(10)), result_scores=list(range(10)))]),
            quality=artifact("quality.json", {"query_count": 1, "ndcg@10": 0.5}))
        self.cost = dict(identity=copy.deepcopy(self.expected),
                         candidate_scope="query_global_unique", rerank_per_cluster=0)

    def check(self):
        return assess_alignment(self.expected, self.quality, self.cost)

    def test_missing_execution_inputs_block_even_matching_reference(self):
        result = self.check()
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["reference_compatible"])
        self.assertFalse(result["integration_ready"])
        self.assertIn("global candidate barrier executor not implemented", result["blockers"])

    def test_per_cluster_is_not_query_r64(self):
        for field, value in (("candidate_scope", "per_cluster"), ("rerank_per_cluster", 64)):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.cost)
                self.cost[field] = value
                self.assertFalse(self.check()["reference_compatible"])
                self.cost = saved

    def test_wrong_index_profile_queryhash(self):
        for field in ("index_sha256", "index_version", "profile_sha256", "query_sha256"):
            with self.subTest(field=field):
                saved = self.cost["identity"][field]
                self.cost["identity"][field] = "5" * 64
                self.assertFalse(self.check()["reference_compatible"])
                self.cost["identity"][field] = saved

    def test_missing_quality_count_never_defaults_zero(self):
        del self.quality["query_count"]
        self.assertFalse(self.check()["reference_compatible"])
        self.quality = None
        self.assertIn("quality evidence missing", self.check()["blockers"])

    def test_missing_quality_data(self):
        Path(self.quality["quality"]["path"]).unlink()
        self.assertFalse(self.check()["reference_compatible"])

    def test_missing_metric_is_blocked_not_zero(self):
        self.quality["quality"] = self.artifact("no-metric.json", {"query_count": 1})
        self.assertFalse(self.check()["reference_compatible"])

    def test_saved_query_hash_mismatch(self):
        row = json.loads(Path(self.quality["rows"]["path"]).read_text())[0]
        row["query_sha256"] = "6" * 64
        self.quality["rows"] = self.artifact("wrong-query.json", [row])
        self.assertFalse(self.check()["reference_compatible"])

    def test_wrong_artifact_hash(self):
        self.quality["rows"]["sha256"] = "0" * 64
        self.assertFalse(self.check()["reference_compatible"])

    def test_duplicate_or_incorrect_candidate_count(self):
        for ids in (list(range(63)), [0] * 64):
            self.quality["rows"] = self.artifact("bad.json", [dict(query_id="q",
                query_sha256="2" * 64, candidate_ids=ids, candidate_scores=[0] * len(ids),
                result_ids=list(range(10)), result_scores=[0] * 10)])
            self.assertFalse(self.check()["reference_compatible"])

    def test_actual_pq_only_report_cannot_pass_as_rerank(self):
        self.cost = {"candidate_scope": "PQ-only modeled scheduling not reranked-quality cost"}
        self.assertFalse(self.check()["reference_compatible"])


if __name__ == "__main__":
    unittest.main()
