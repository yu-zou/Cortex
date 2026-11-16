#!/usr/bin/env python3
"""End-to-end interop: Python adapter CSV -> compiled C++ selector runner.

Uses only the standard library and a small synthetic two-query/four-subtask
fixture. No real FiQA data, no system performance claim.
"""
import csv
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

MODEL_ROOT = pathlib.Path(__file__).resolve().parents[1]
ADAPTER = MODEL_ROOT / "scripts" / "fiqa_to_scheduler_trace.py"
RUNNER = pathlib.Path(os.environ.get(
    "SCHEDULER_TRACE_RUNNER",
    "/tmp/opencode/cortex-scheduler-trace-build/scheduler_trace_runner",
))

sys.path.insert(0, str(MODEL_ROOT / "scripts"))
from fiqa_to_scheduler_trace import write_scheduler_csv

CSV_HEADER = [
    "trace", "seed", "task_id", "query_id", "arrival_ns", "service_ns",
    "estimate_ns", "estimate_label", "task_class", "query_class",
]
ARMS = {"fifo", "sjf_no_aging", "sjf_aging_2ms", "query_aware_2ms"}
SERVICES = [12112, 13112, 12112, 13112]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task(query_id=1, task_class="cluster_0", query_class="nprobe2_cands110"):
    return {
        "task_id": 1,
        "query_id": query_id,
        "arrival_ns": 0,
        "service_ns": 12112,
        "estimate_ns": 12112,
        "task_class": task_class,
        "query_class": query_class,
    }


class AdapterCsvWriterTests(unittest.TestCase):
    """CSV writer quoting contract shared with the C++ line reader."""

    def test_comma_and_quote_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "quoted.csv"
            write_scheduler_csv(
                [task(task_class="cluster,0", query_class='q"uoted')], path)
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(rows[0], CSV_HEADER)
        self.assertEqual(len(rows[1]), 10)
        self.assertEqual(rows[1][8], "cluster,0")
        self.assertEqual(rows[1][9], 'q"uoted')

    def test_newline_rejected_before_file_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "multiline.csv"
            with self.assertRaisesRegex(ValueError, "unsupported newline"):
                write_scheduler_csv([task(task_class="bad\nclass")], path)
            self.assertFalse(path.exists())


class AdapterToCppInteropTests(unittest.TestCase):
    """Run the actual adapter CLI, then the compiled C++ runner."""

    @classmethod
    def setUpClass(cls):
        if not RUNNER.is_file():
            raise AssertionError(
                "compiled runner not found: {} (set SCHEDULER_TRACE_RUNNER)"
                .format(RUNNER))

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.write_fixture()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_query_atomic_cli_and_staggered_preflight(self):
        self.run_adapter()
        output = self.root / "atomic"
        output.mkdir()
        command = [str(RUNNER), "--input-csv", str(self.csv_path),
                   "--admission", "query_atomic", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / "scheduler-trace-queries.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["arm"] for row in rows}, ARMS)
        self.assertTrue(all(row["status"] == "completed" for row in rows))
        with self.csv_path.open() as stream:
            inputs = list(csv.DictReader(stream))
        inputs[1]["arrival_ns"] = "1"
        with self.csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(inputs)
        empty = self.root / "staggered"
        empty.mkdir()
        command[-1] = str(empty)
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("staggered query membership", result.stderr)
        self.assertEqual(list(empty.iterdir()), [])

    def test_offline_capacity_cli(self):
        self.run_adapter()
        for value in ("0", "1", "33", "256", "-1", "64x", "064", "",
                      "18446744073709551616"):
            with self.subTest(invalid=value):
                out = self.root / ("bad" + str(len(list(self.root.iterdir()))))
                out.mkdir()
                result = subprocess.run([str(RUNNER), "--input-csv", str(self.csv_path),
                                         "--capacity", value, str(out)],
                                        capture_output=True, text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(out.iterdir()), [])
        for capacity in (32, 64, 128):
            for atomic in (False, True):
                out = self.root / ("cap{}-{}".format(capacity, atomic))
                out.mkdir()
                command = [str(RUNNER), "--input-csv", str(self.csv_path),
                           "--capacity", str(capacity)]
                if atomic:
                    command += ["--admission", "query_atomic"]
                result = subprocess.run(command + [str(out)], capture_output=True,
                                        text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads((out / "scheduler-traces.json").read_text())
                self.assertEqual(data["model"]["system_capacity_active_plus_queued"], capacity)
                self.assertEqual(data["model"]["scheduler_capacity"], capacity)
                with (out / "scheduler-trace-summary.csv").open() as stream:
                    summaries = list(csv.DictReader(stream))
                self.assertTrue(all(int(r["system_capacity_active_plus_queued"]) == capacity
                                    for r in summaries))

    def write_fixture(self):
        metadata = {
            "index_namespace": "interop/synthetic",
            "index_version": "v1",
            "dataset": "synthetic",
            "index_id": "fixture",
            "encoder_model": "sentence-transformers/all-mpnet-base-v2",
            "encoder_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "metric": "IP",
            "normalized": True,
            "split": "test",
            "total_centroids": 2,
            "nprobe": 2,
        }
        encoder = {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "normalize_embeddings": True,
        }
        probes = [
            {"query_id": "q1", "query_row": 0, "nprobe": 2,
             "centroid_ids": [0, 1], "cluster_sizes": [10, 100],
             "candidate_count": 110},
            {"query_id": "q2", "query_row": 1, "nprobe": 2,
             "centroid_ids": [0, 1], "cluster_sizes": [10, 100],
             "candidate_count": 110},
        ]
        self.metadata_path = self.inputs / "metadata.json"
        self.encoder_path = self.inputs / "encoder.json"
        self.sizes_path = self.inputs / "cluster_sizes.json"
        self.probes_path = self.inputs / "probes.jsonl"
        self.csv_path = self.inputs / "trace.csv"
        self.manifest_path = self.inputs / "manifest.json"
        self.metadata_path.write_text(json.dumps(metadata))
        self.encoder_path.write_text(json.dumps(encoder))
        self.sizes_path.write_text(json.dumps([10, 100]))
        self.probes_path.write_text(
            "\n".join(json.dumps(probe) for probe in probes) + "\n")

    def run_adapter(self):
        command = [
            sys.executable, str(ADAPTER),
            "--probes", str(self.probes_path),
            "--metadata", str(self.metadata_path),
            "--cluster-sizes", str(self.sizes_path),
            "--encoder-config", str(self.encoder_path),
            "--expected-queries", "2",
            "--output-csv", str(self.csv_path),
            "--output-manifest", str(self.manifest_path),
            "--rate-qps", "1.0",
            "--start-ns", "0",
            "--clock-ns", "1000/180",
            "--fixed-cycles", "2160",
            "--cycles-per-doc", "2",
        ]
        result = subprocess.run(
            command, cwd=str(MODEL_ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def run_runner(self, arguments):
        return subprocess.run(
            [str(RUNNER)] + arguments, capture_output=True, text=True,
            timeout=120)

    def read_csv(self, path):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, CSV_HEADER)
            return list(reader)

    def test_adapter_csv_to_cpp_selector(self):
        self.run_adapter()
        input_hash = sha256(self.csv_path)
        adapter_rows = self.read_csv(self.csv_path)
        self.assertEqual(len(adapter_rows), 4)
        self.assertEqual([int(row["service_ns"]) for row in adapter_rows], SERVICES)
        self.assertEqual([int(row["query_id"]) for row in adapter_rows],
                         [1, 1, 2, 2])
        self.assertEqual(
            [row["estimate_label"] for row in adapter_rows],
            ["trace_supplied"] * 4)

        output_dir = self.root / "out"
        output_dir.mkdir()
        result = self.run_runner([
            "--input-csv", str(self.csv_path), str(output_dir)])
        self.assertEqual(result.returncode, 0, result.stderr)

        # The runner must not modify its input.
        self.assertEqual(sha256(self.csv_path), input_hash)
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {
                "csv-load-manifest.txt",
                "scheduler-trace-inputs.csv",
                "scheduler-trace-tasks.csv",
                "scheduler-trace-queries.csv",
                "scheduler-trace-summary.csv",
                "scheduler-traces.json",
            },
        )

        runner_inputs = self.read_csv(
            output_dir / "scheduler-trace-inputs.csv")
        self.assertEqual(len(runner_inputs), 4)
        self.assertEqual([int(row["task_id"]) for row in runner_inputs],
                         [1, 2, 3, 4])
        self.assertEqual([int(row["query_id"]) for row in runner_inputs],
                         [1, 1, 2, 2])
        self.assertEqual([int(row["service_ns"]) for row in runner_inputs],
                         SERVICES)
        self.assertEqual(
            [row["estimate_label"] for row in runner_inputs],
            ["trace_supplied"] * 4)

        with (output_dir / "scheduler-trace-tasks.csv").open(newline="") as stream:
            task_rows = list(csv.DictReader(stream))
        self.assertEqual(len(task_rows), 16)
        self.assertEqual({row["arm"] for row in task_rows}, ARMS)
        for arm in ARMS:
            arm_rows = [row for row in task_rows if row["arm"] == arm]
            self.assertEqual(len(arm_rows), 4)
            self.assertEqual([int(row["query_id"]) for row in arm_rows],
                             [1, 1, 2, 2])
            self.assertEqual([int(row["service_ns"]) for row in arm_rows],
                             SERVICES)
            self.assertTrue(all(row["outcome"] == "completed"
                                for row in arm_rows))

        with (output_dir / "scheduler-trace-queries.csv").open(newline="") as stream:
            query_rows = list(csv.DictReader(stream))
        self.assertEqual(len(query_rows), 8)
        for arm in ARMS:
            arm_rows = [row for row in query_rows if row["arm"] == arm]
            self.assertEqual({int(row["query_id"]) for row in arm_rows}, {1, 2})
            for row in arm_rows:
                self.assertEqual(int(row["subtasks"]), 2)
                self.assertEqual(int(row["completed_subtasks"]), 2)
                self.assertEqual(int(row["rejected_subtasks"]), 0)
                self.assertEqual(row["status"], "completed")

        with (output_dir / "scheduler-trace-summary.csv").open(newline="") as stream:
            summary_rows = list(csv.DictReader(stream))
        self.assertEqual(len(summary_rows), 4)
        self.assertEqual({row["arm"] for row in summary_rows}, ARMS)
        for row in summary_rows:
            self.assertEqual(int(row["tasks_arrived"]), 4)
            self.assertEqual(int(row["tasks_completed"]), 4)
            self.assertEqual(int(row["tasks_rejected"]), 0)
            self.assertEqual(int(row["tasks_pending"]), 0)
            self.assertLessEqual(int(row["max_occupancy"]), 32)
            self.assertEqual(
                int(row["tasks_arrived"]),
                int(row["tasks_completed"]) + int(row["tasks_rejected"]) +
                int(row["tasks_pending"]))

        data = json.loads((output_dir / "scheduler-traces.json").read_text())
        self.assertEqual(len(data["runs"]), 4)
        for run in data["runs"]:
            tasks = run["tasks"]
            self.assertEqual(tasks["arrived"], 4)
            self.assertEqual(tasks["completed"], 4)
            self.assertEqual(tasks["rejected"], 0)
            self.assertEqual(tasks["pending"], 0)
            self.assertEqual(tasks["arrived"],
                             tasks["completed"] + tasks["rejected"] +
                             tasks["pending"])

    def test_external_quoted_text_roundtrip(self):
        trace_text = 'synthetic,trace "quoted" \\ path\tcontrol'
        task_class = 'class,with "quote" \\ slash\tcontrol'
        query_class_1 = 'query,class "q1" \\ slash\tcontrol'
        query_class_2 = 'query,class "q2" \\ slash\tcontrol'
        input_path = self.root / "quoted-input.csv"
        rows = [
            [trace_text, 0, 1, 1, 0, 12112, 12112, "trace_supplied",
             task_class, query_class_1],
            [trace_text, 0, 2, 1, 0, 13112, 13112, "trace_supplied",
             task_class, query_class_1],
            [trace_text, 0, 3, 2, 100000, 12112, 12112, "trace_supplied",
             task_class, query_class_2],
            [trace_text, 0, 4, 2, 100000, 13112, 13112, "trace_supplied",
             task_class, query_class_2],
        ]
        with input_path.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        input_hash = sha256(input_path)

        output_dir = self.root / "quoted-out"
        output_dir.mkdir()
        result = self.run_runner([
            "--input-csv", str(input_path), str(output_dir)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sha256(input_path), input_hash)

        expected_columns = {
            "scheduler-trace-inputs.csv": 10,
            "scheduler-trace-tasks.csv": 14,
            "scheduler-trace-queries.csv": 12,
            "scheduler-trace-summary.csv": 24,
        }
        for name, columns in expected_columns.items():
            with (output_dir / name).open(newline="") as stream:
                parsed = list(csv.reader(stream))
            self.assertEqual(len(parsed[0]), columns, name)
            for row in parsed:
                self.assertEqual(len(row), columns, (name, row))

        with (output_dir / "scheduler-trace-inputs.csv").open(newline="") as stream:
            inputs = list(csv.DictReader(stream))
        self.assertEqual(len(inputs), 4)
        for row in inputs:
            self.assertEqual(row["trace"], trace_text)
            self.assertEqual(row["task_class"], task_class)
        self.assertEqual([row["query_class"] for row in inputs],
                         [query_class_1, query_class_1,
                          query_class_2, query_class_2])
        self.assertEqual([int(row["query_id"]) for row in inputs],
                         [1, 1, 2, 2])
        self.assertEqual([int(row["service_ns"]) for row in inputs],
                         [12112, 13112, 12112, 13112])

        with (output_dir / "scheduler-trace-tasks.csv").open(newline="") as stream:
            tasks = list(csv.DictReader(stream))
        self.assertEqual(len(tasks), 16)
        for row in tasks:
            self.assertEqual(row["trace"], trace_text)
            self.assertEqual(row["task_class"], task_class)
            self.assertIn(row["arm"], ARMS)
            self.assertEqual(row["outcome"], "completed")
        for arm in ARMS:
            arm_rows = [row for row in tasks if row["arm"] == arm]
            self.assertEqual([int(row["query_id"]) for row in arm_rows],
                             [1, 1, 2, 2])
            self.assertEqual([int(row["service_ns"]) for row in arm_rows],
                             [12112, 13112, 12112, 13112])

        with (output_dir / "scheduler-trace-queries.csv").open(newline="") as stream:
            queries = list(csv.DictReader(stream))
        self.assertEqual(len(queries), 8)
        expected_query_text = {1: query_class_1, 2: query_class_2}
        for row in queries:
            self.assertEqual(row["trace"], trace_text)
            self.assertEqual(row["query_class"],
                             expected_query_text[int(row["query_id"])])
            self.assertIn(row["arm"], ARMS)
            self.assertEqual(int(row["subtasks"]), 2)
            self.assertEqual(int(row["completed_subtasks"]), 2)
            self.assertEqual(row["status"], "completed")
        for arm in ARMS:
            arm_rows = [row for row in queries if row["arm"] == arm]
            self.assertEqual({int(row["query_id"]) for row in arm_rows}, {1, 2})

        with (output_dir / "scheduler-trace-summary.csv").open(newline="") as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual(len(summary), 12)
        self.assertEqual({row["query_class"] for row in summary},
                         {"all", query_class_1, query_class_2})
        for row in summary:
            self.assertEqual(row["trace"], trace_text)
            self.assertIn(row["arm"], ARMS)
            expected_queries = 2 if row["query_class"] == "all" else 1
            self.assertEqual(int(row["queries"]), expected_queries)
            self.assertEqual(int(row["completed"]), expected_queries)
            self.assertEqual(int(row["tasks_arrived"]), 4)
            self.assertEqual(int(row["tasks_completed"]), 4)
            self.assertEqual(int(row["tasks_rejected"]), 0)
            self.assertEqual(int(row["tasks_pending"]), 0)

        data = json.loads((output_dir / "scheduler-traces.json").read_text())
        self.assertEqual({arm["name"] for arm in data["arms"]}, ARMS)
        self.assertEqual(len(data["runs"]), 4)
        for run in data["runs"]:
            self.assertEqual(run["trace"], trace_text)
            self.assertIn(run["arm"], ARMS)
            classes = {entry["class"]: entry for entry in run["queries"]}
            self.assertEqual(set(classes), {"all", query_class_1, query_class_2})
            self.assertEqual(classes["all"]["queries"], 2)
            self.assertEqual(classes[query_class_1]["queries"], 1)
            self.assertEqual(classes[query_class_2]["queries"], 1)
            self.assertEqual(run["tasks"]["arrived"], 4)
            self.assertEqual(run["tasks"]["completed"], 4)
            self.assertEqual(run["tasks"]["rejected"], 0)
            self.assertEqual(run["tasks"]["pending"], 0)

    def test_wrong_argc_rejected_without_outputs(self):
        output_dir = self.root / "out"
        output_dir.mkdir()
        cases = [
            [str(RUNNER), "--input-csv"],
            [str(RUNNER), "--input-csv", str(self.csv_path)],
            [str(RUNNER), "--input-csv", str(self.csv_path), str(output_dir),
             "extra"],
            [str(RUNNER), str(self.csv_path), str(output_dir)],
        ]
        for command in cases:
            with self.subTest(command=command[1:]):
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(output_dir.iterdir()), [])

    def assert_loader_rejects_before_output(self, content, name):
        bad_csv = self.root / (name + ".csv")
        bad_csv.write_text(content)
        output_dir = self.root / (name + "-out")
        output_dir.mkdir()
        result = self.run_runner([
            "--input-csv", str(bad_csv), str(output_dir)])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(output_dir.iterdir()), [])

    def test_malformed_input_rejected_before_outputs(self):
        header = ",".join(CSV_HEADER) + "\n"
        valid = "test,0,1,1,0,100,100,trace_supplied,cls,qcls\n"
        cases = [
            ("bad-header", "wrong,header,columns\n"),
            ("bad-query-id",
             header + "test,0,1,not_numeric,0,100,100,trace_supplied,cls,qcls\n"),
            ("bad-label",
             header + "test,0,1,1,0,100,100,wrong_label,cls,qcls\n"),
            ("duplicate-id",
             header + valid +
             "test,0,1,2,1000,100,100,trace_supplied,cls,qcls\n"),
        ]
        for name, content in cases:
            with self.subTest(case=name):
                self.assert_loader_rejects_before_output(content, name)

    def test_nonempty_output_rejected_and_input_preserved(self):
        self.run_adapter()
        input_hash = sha256(self.csv_path)
        output_dir = self.root / "out"
        output_dir.mkdir()
        sentinel = output_dir / "sentinel.txt"
        sentinel.write_text("keep")
        result = self.run_runner([
            "--input-csv", str(self.csv_path), str(output_dir)])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual({path.name for path in output_dir.iterdir()},
                         {"sentinel.txt"})
        self.assertEqual(sha256(self.csv_path), input_hash)


if __name__ == "__main__":
    unittest.main()
