import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.probe_contract import IndexMetadata, MPNET_REVISION, load_probes
from scripts.real_probe_des import adapt_probes, load_input, run_policies


def probes(namespace="a", version="v1", ids=(2, 0), qid="q"):
    meta = IndexMetadata(namespace, version, "fiqa", "ivfpq8", MPNET_REVISION,
                         768, "IP", True, "test", 3, 2)
    sizes = [3, 0, 7]
    row = dict(query_id=qid, query_row=0, centroid_ids=list(ids),
               cluster_sizes=[sizes[i] for i in ids],
               candidate_count=sum(sizes[i] for i in ids))
    return load_probes(io.StringIO(json.dumps(row)), meta, sizes)


class AdapterTests(unittest.TestCase):
    def test_standalone_clean_subprocess(self):
        model_root = Path(__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONHOME", "SCHED_STATS_PATH")
               and not k.startswith("OSD_")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = """
import pathlib, sys
repo = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
from scripts.real_probe_des import adapt_probes, run_policies
assert 'scripts.validate_model' not in sys.modules
assert 'scripts.extract_hls' not in sys.modules
for name, module in tuple(sys.modules.items()):
    if name.startswith(('scripts.', 'strata_sim.')) and getattr(module, '__file__', None):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(repo), (name, module.__file__)
print('repo-only import PASS')
"""
            imported = subprocess.run([sys.executable, "-I", "-c", code, str(model_root)],
                                      cwd=root, env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(imported.returncode, 0, imported.stderr)
            p = probes()[0]
            (root / "metadata.json").write_text(json.dumps(p.metadata.__dict__))
            (root / "sizes.json").write_text("[3,0,7]")
            (root / "encoder.json").write_text(json.dumps(dict(
                model=p.metadata.encoder_model, revision=p.metadata.encoder_revision,
                dimension=768, normalize_embeddings=True)))
            (root / "probes.jsonl").write_text(json.dumps(dict(query_id=p.query_id,
                centroid_ids=list(p.centroid_ids), cluster_sizes=list(p.cluster_sizes),
                candidate_count=p.candidate_count)) + "\n")
            output = root / "result.json"
            command = [sys.executable, "-I", str(model_root / "scripts/real_probe_des.py"),
                "--probes", str(root / "probes.jsonl"), "--metadata", str(root / "metadata.json"),
                "--cluster-sizes", str(root / "sizes.json"), "--encoder-config", str(root / "encoder.json"),
                "--expected-queries", "1", "--rate-qps", "1000", "--slo-ns", "12345",
                "--policies", "fifo", "--output", str(output)]
            executed = subprocess.run(command, cwd=root, env=env, text=True,
                                      capture_output=True, timeout=20)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["policies"]["fifo"]["counters"]["subqueries"], 2)
            self.assertEqual(result["candidate_scope"], "PQ-only modeled scheduling not reranked-quality cost")
            for module in result["loaded_modules"].values():
                self.assertTrue(Path(module["path"]).resolve().is_relative_to(model_root))

    def adapt(self, rows):
        return adapt_probes(rows, rate_qps=1000, slo_ns=12345, top_k=10)

    def test_namespace_and_version_isolation(self):
        trace = self.adapt(probes() + probes("b") + probes(version="v2"))
        self.assertEqual(len(trace.cluster_sizes), 6)
        self.assertEqual(len({q.query_id for q in trace.queries}), 3)
        self.assertEqual(len({q.codebook_id for q in trace.queries}), 3)
        self.assertEqual(len({c for q in trace.queries for c in q.cluster_ids}), 6)

    def test_exact_probes_order_and_sizes(self):
        trace = self.adapt(probes())
        q = trace.queries[0]
        self.assertEqual(len(q.cluster_ids), 2)
        self.assertEqual([trace.cluster_sizes[c] for c in q.cluster_ids], [7, 3])
        self.assertEqual(trace.references[q.query_id]["candidate_count"], 10)

    def test_uniform_arrival_and_slo_budget(self):
        trace = self.adapt(probes(qid="q1") + probes(qid="q2") + probes(qid="q3"))
        self.assertEqual([q.arrival_ns for q in trace.queries], [0, 1000000, 2000000])
        self.assertTrue(all(q.deadline_ns - q.arrival_ns == 12345 for q in trace.queries))

    def test_mapping_independent_of_input_order(self):
        rows = probes() + probes("b")
        a, b = self.adapt(rows), self.adapt(tuple(reversed(rows)))
        self.assertEqual(a.cluster_mapping, b.cluster_mapping)
        self.assertEqual(a.cluster_sizes, b.cluster_sizes)
        self.assertEqual({q.query_id: q.cluster_ids for q in a.queries},
                         {q.query_id: q.cluster_ids for q in b.queries})

    def test_zero_size_cluster_retained(self):
        trace = self.adapt(probes(ids=(1, 0)))
        self.assertEqual([trace.cluster_sizes[c] for c in trace.queries[0].cluster_ids], [0, 3])
        report = run_policies(trace, policies=("fifo",), num_osds=3)
        self.assertEqual(report["policies"]["fifo"]["counters"]["subqueries"], 2)

    def test_invalid_file_and_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = probes()[0].metadata
            (root / "metadata.json").write_text(json.dumps(meta.__dict__))
            (root / "sizes.json").write_text("[3,0,7]")
            encoder = dict(model=meta.encoder_model, revision=meta.encoder_revision,
                           dimension=768, normalize_embeddings=True)
            (root / "encoder.json").write_text(json.dumps(encoder))
            (root / "probes.jsonl").write_text('{"query_id": "bad"}\n')
            args = [root / n for n in ("probes.jsonl", "metadata.json", "sizes.json", "encoder.json")]
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_input(*args, expected_queries=1)
            encoder["revision"] = "0" * 40
            (root / "encoder.json").write_text(json.dumps(encoder))
            with self.assertRaisesRegex(ValueError, "encoder"):
                load_input(*args, expected_queries=1)

    def test_invalid_adaptation(self):
        for rate in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                adapt_probes(probes(), rate_qps=rate, slo_ns=1, top_k=10)
        for rows in ((), probes() + probes(),
                     (replace(probes()[0], candidate_count=11),),
                     (replace(probes()[0], centroid_ids=(2,)),)):
            with self.assertRaises(ValueError):
                self.adapt(rows)

    def test_real_execution_pq_only_and_identity_across_policies(self):
        trace = self.adapt(probes(qid="q1") + probes(qid="q2"))
        report = run_policies(trace, policies=("fifo", "lstf", "multifactor"), num_osds=3)
        self.assertEqual(report["candidate_scope"], "PQ-only modeled scheduling not reranked-quality cost")
        identities = []
        for result in report["policies"].values():
            self.assertEqual(result["counters"]["completed"], 2)
            self.assertEqual(result["counters"]["subqueries"], 4)
            self.assertEqual(result["counters"]["raw_transfers"], 0)
            self.assertEqual(result["counters"]["cpu_reservations"], 0)
            for row in result["requests"]:
                self.assertEqual(row["finish_ns"], max(t["end_ns"] for t in row["subqueries"]))
                self.assertEqual(len(row["subqueries"]), 2)
                self.assertEqual(row["oracle_reference"], trace.references[row["query_id"]])
            identities.append([r["query_id"] for r in result["requests"]])
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[0], identities[2])

    def test_partial_completion_refused(self):
        from scripts import real_probe_des as adapter
        actual = adapter.run_simulation

        def broken(*args, **kwargs):
            completed, info = actual(*args, **kwargs)
            completed[0].timings = completed[0].timings[:-1]
            return completed, info

        with patch.object(adapter, "run_simulation", broken):
            with self.assertRaisesRegex(ValueError, "barrier"):
                run_policies(self.adapt(probes()), policies=("fifo",), num_osds=3)


if __name__ == "__main__":
    unittest.main()
