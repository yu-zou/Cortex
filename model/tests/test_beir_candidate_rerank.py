"""Synthetic checks for standalone CPU candidate reranking (no model downloads)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve()
SCRIPT = HERE.parent.parent / "scripts" / "beir_candidate_rerank.py"
if not SCRIPT.is_file():
    SCRIPT = HERE.with_name("beir_candidate_rerank.py")
SPEC = importlib.util.spec_from_file_location("candidate_rerank", SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class CandidateRerankTests(unittest.TestCase):
    def test_exact_dot_and_ties(self):
        rng = np.random.default_rng(6)
        docs = rng.normal(size=(40, 7)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1)[:, None]
        query = rng.normal(size=7).astype(np.float32)
        query /= np.linalg.norm(query)
        ids = np.arange(32)[::-1]
        rows, scores = pilot.rerank(query, docs, ids)
        independent = np.array([np.dot(docs[i].astype(float), query.astype(float)) for i in ids])
        order = sorted(range(len(ids)), key=lambda i: (-independent[i], int(ids[i])))[:10]
        np.testing.assert_array_equal(rows, ids[order])
        np.testing.assert_allclose(scores, independent[order], atol=1e-6)
        tied = np.tile(np.array([1., 0.], dtype=np.float32), (20, 1))
        rows, _ = pilot.rerank(tied[0], tied, np.arange(20)[::-1])
        np.testing.assert_array_equal(rows, np.arange(10))

    def test_duplicates_deduplicated_before_scoring(self):
        docs = np.eye(12, dtype=np.float32)
        rows, _ = pilot.rerank(docs[0], docs, [0, 0] + list(range(12)))
        self.assertEqual(len(set(rows)), 10)
        np.testing.assert_array_equal(rows, np.arange(10))

    def test_invalid_candidates(self):
        for ids in ([-1] + list(range(10)), list(range(10)) + [20],
                    list(range(9)), [0] * 32, np.arange(10, dtype=float)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                pilot.candidate_ids(ids, 20)

    def test_normalization_and_nonfinite_rejected(self):
        for matrix in (np.zeros((10, 3), dtype=np.float32),
                       np.full((10, 3), np.nan, dtype=np.float32), np.eye(10)):
            with self.assertRaises(ValueError):
                pilot.validate_vectors(matrix)
        with self.assertRaises(ValueError):
            pilot.rank(np.arange(10), np.full(10, np.nan))

    def test_mapping_duplicates_and_out_of_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mapping.jsonl"
            for rows in ([{"row": 0, "document_id": "a"}, {"row": 1, "document_id": "a"}],
                         [{"row": 1, "document_id": "a"}, {"row": 0, "document_id": "b"}],
                         [{"row": 0, "document_id": ""}, {"row": 1, "document_id": "b"}]):
                path.write_text("\n".join(map(json.dumps, rows)))
                with self.assertRaises(ValueError):
                    pilot.load_mapping(path, "document_id", 2)

    def test_hand_computed_metrics_and_qrels_independence(self):
        rows = np.arange(10)[None, :]
        dids = [str(i) for i in range(12)]
        qrels = {"q": {"0": 2, "1": 1, "11": 1}}
        metrics = pilot.evaluate(rows, rows, dids, ["q"], qrels)
        dcg = 2 + 1 / np.log2(3)
        self.assertAlmostEqual(metrics["ndcg@10"], dcg / (dcg + .5))
        self.assertAlmostEqual(metrics["recall@10"], 2 / 3)
        self.assertEqual(metrics["exact_overlap@10"], 1.)
        with self.assertRaises(ValueError):
            pilot.evaluate(np.zeros((1, 10), dtype=int), rows, dids, ["q"], qrels)
        # Retrieval API accepts no relevance data; evaluation cannot mutate rows.
        np.testing.assert_array_equal(rows, np.arange(10)[None, :])

    def test_qrels_invalid_and_duplicate_references(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.tsv"
            for body in ("q\tmissing\t1\n", "q\td\t1\nq\td\t1\n", "q\td\t0\n"):
                path.write_text("query-id\tcorpus-id\tscore\n" + body)
                with self.assertRaises(ValueError):
                    pilot.load_qrels(path, ["q"], ["d"])

    def test_exclusive_writes_and_existing_run_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            pilot.write_json(path, {"original": True})
            with self.assertRaises(FileExistsError):
                pilot.write_json(path, {"changed": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})
            with self.assertRaises(FileExistsError):
                pilot.run(Path(folder), Path(folder))


if __name__ == "__main__":
    unittest.main(verbosity=2)
