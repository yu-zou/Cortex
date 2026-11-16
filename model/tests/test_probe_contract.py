"""Strict probe input contract; no scheduler or third-party dependencies."""
import dataclasses
import io
import json
import pathlib
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from probe_contract import ClusterIdentity, IndexMetadata, load_probes


class ProbeContractTests(unittest.TestCase):
    def metadata(self, **changes):
        values = dict(index_namespace="beir/fiqa", index_version="sha256:pinned",
                      dataset="fiqa", index_id="ivfpq8", encoder_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
                      dimension=768, metric="IP", normalized=True, split="test",
                      total_centroids=64, nprobe=32)
        values.update(changes)
        return IndexMetadata(**values)

    def row(self):
        return dict(query_id="10034", query_row=0, nprobe=32,
                    centroid_ids=list(range(32)), cluster_sizes=list(range(32)),
                    candidate_count=sum(range(32)))

    def load(self, rows=None, metadata=None, sizes=None):
        rows = [self.row()] if rows is None else rows
        return load_probes(io.StringIO("".join(json.dumps(r) + "\n" for r in rows)),
                           metadata or self.metadata(), tuple(range(64)) if sizes is None else sizes)

    def test_real_shape_and_order(self):
        row = self.row()
        row["centroid_ids"].reverse()
        row["cluster_sizes"].reverse()
        probes = self.load([row])
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0].centroid_ids, tuple(reversed(range(32))))
        self.assertEqual(probes[0].cluster_identities[0], ClusterIdentity("beir/fiqa", "sha256:pinned", 31))
        self.assertIsInstance(probes, tuple)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            probes[0].query_id = "other"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            probes[0].metadata.dimension = 5

    def test_collision_proof_identity(self):
        identities = [ClusterIdentity("a/b", "c", 0), ClusterIdentity("a", "b/c", 0),
                      ClusterIdentity("a/b", "c2", 0), ClusterIdentity("a/b", "c", 1)]
        self.assertEqual(len(set(identities)), 4)
        self.assertEqual(len({i.key for i in identities}), 4)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identities[0].index_version = "new"

    def test_reject_bad_rows(self):
        mutations = [lambda r: r.pop("query_id"), lambda r: r.update(query_id=""),
                     lambda r: r.update(query_id=123), lambda r: r.update(candidate_count=-1),
                     lambda r: r.update(candidate_count=True), lambda r: r.update(candidate_count=1.0),
                     lambda r: r.update(candidate_count=0), lambda r: r.update(nprobe=64),
                     lambda r: r.update(query_row=-1), lambda r: r.update(extra="ignored?"),
                     lambda r: r["centroid_ids"].pop(), lambda r: r["cluster_sizes"].pop(),
                     lambda r: r["centroid_ids"].__setitem__(0, 1),
                     lambda r: r["centroid_ids"].__setitem__(0, 64),
                     lambda r: r["centroid_ids"].__setitem__(0, -1),
                     lambda r: r["centroid_ids"].__setitem__(0, True),
                     lambda r: r["cluster_sizes"].__setitem__(0, -1),
                     lambda r: r["cluster_sizes"].__setitem__(0, 1)]
        for mutate in mutations:
            row = self.row()
            mutate(row)
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.load([row])

    def test_duplicates_and_empty(self):
        with self.assertRaisesRegex(ValueError, "duplicate query"):
            self.load([self.row(), self.row()])
        with self.assertRaises(ValueError):
            self.load([])

    def test_json_strictness(self):
        for text in ['\n', '{}\n\n', '[]\n', '{bad}\n',
                     '{"query_id":"a","query_id":"b"}\n', '{"x":NaN}\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_probes(io.StringIO(text), self.metadata(), tuple(range(64)))

    def test_metadata_validation(self):
        for changes in [dict(index_namespace=""), dict(index_version=" "), dict(split=""),
                        dict(encoder_revision="main"), dict(dimension=True), dict(dimension=0),
                        dict(metric="cosine"), dict(normalized=1), dict(total_centroids=32, nprobe=33)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.metadata(**changes)

    def test_dimension_pin_and_size_table(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.metadata(dimension=384)
        for sizes in [list(range(63)), list(range(63)) + [-1], list(range(63)) + [True]]:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                self.load(sizes=sizes)

    def test_optional_pilot_fields(self):
        row = self.row()
        del row["query_row"]
        del row["nprobe"]
        self.assertIsNone(self.load([row])[0].query_row)

    def test_cli_pins_encoder_and_count(self):
        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "probe_contract.py"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            metadata = dataclasses.asdict(self.metadata())
            encoder = dict(model=metadata["encoder_model"], revision=metadata["encoder_revision"],
                           dimension=768, normalize_embeddings=True)
            for name, value in [("metadata", metadata), ("encoder", encoder), ("sizes", list(range(64)))]:
                (root / (name + ".json")).write_text(json.dumps(value))
            (root / "probes.jsonl").write_text(json.dumps(self.row()) + "\n")
            command = [sys.executable, str(script), str(root / "probes.jsonl"),
                       "--metadata", str(root / "metadata.json"),
                       "--cluster-sizes", str(root / "sizes.json"),
                       "--encoder-config", str(root / "encoder.json"), "--expected-queries", "1"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["query_count"], 1)
            for field, value in [("revision", "main"), ("dimension", 384),
                                 ("normalize_embeddings", False), ("model", "other")]:
                changed = dict(encoder, **{field: value})
                (root / "encoder.json").write_text(json.dumps(changed))
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("encoder_config mismatch", result.stderr)
            (root / "encoder.json").write_text(json.dumps(encoder))
            result = subprocess.run(command[:-1] + ["2"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unexpected query count", result.stderr)


if __name__ == "__main__":
    unittest.main()
