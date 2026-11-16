#!/usr/bin/env python3
"""Numeric regression tests for FiQA-to-scheduler trace adapter.

Tests rational arithmetic, overflow detection, parameter validation, and CLI
contract using actual Probe/IndexMetadata fixtures with unequal cluster sizes.
Uses only standard library unittest; no pytest dependency.
"""
import csv
import io
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from fiqa_to_scheduler_trace import (
    UINT64_MAX,
    build_query_id_mapping,
    compute_cluster_service_cost,
    generate_trace,
    query_identity,
    validate_inputs,
    write_manifest,
    write_scheduler_csv,
)
from probe_contract import IndexMetadata, Probe


class FiQASchedulerNumericTests(unittest.TestCase):
    """Numeric regression tests with actual Probe fixtures."""
    
    def metadata(self):
        """Return test metadata matching probe_contract conventions."""
        return IndexMetadata(
            index_namespace="beir/fiqa",
            index_version="sha256:pinned",
            dataset="fiqa",
            index_id="ivfpq8",
            encoder_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            dimension=768,
            metric="IP",
            normalized=True,
            split="test",
            total_centroids=2,  # Small fixture for testing
            nprobe=2,
        )
    
    def probe_unequal_clusters(self):
        """Probe with unequal cluster sizes: 10 and 100, total 110."""
        return Probe(
            metadata=self.metadata(),
            query_id="test_query_001",
            centroid_ids=(0, 1),
            cluster_sizes=(10, 100),
            candidate_count=110,
            query_row=0,
        )
    
    def test_exact_rational_service_cost(self):
        """Test exact rational arithmetic: clock 1000/180, fixed 2160, per_doc 2."""
        # Cluster size 10: (2160 + 2*10) * (1000/180) = 2180 * (1000/180) = 12111.111... -> ceil 12112
        cost_10 = compute_cluster_service_cost(
            10, "declared_scan_proxy",
            {"clock_ns": "1000/180", "fixed_cycles": 2160, "cycles_per_doc": 2}
        )
        self.assertEqual(cost_10, 12112)
        
        # Cluster size 100: (2160 + 2*100) * (1000/180) = 2360 * (1000/180) = 13111.111... -> ceil 13112
        cost_100 = compute_cluster_service_cost(
            100, "declared_scan_proxy",
            {"clock_ns": "1000/180", "fixed_cycles": 2160, "cycles_per_doc": 2}
        )
        self.assertEqual(cost_100, 13112)
        
        # Verify ceil behavior: exact integer should not increase
        cost_exact = compute_cluster_service_cost(
            50, "declared_scan_proxy",
            {"clock_ns": 2.0, "fixed_cycles": 50, "cycles_per_doc": 0}
        )
        self.assertEqual(cost_exact, 100)  # 50 * 2.0 = 100
    
    def test_rational_string_preservation(self):
        """Test that rational strings like '1000/180' are preserved, not float-converted."""
        # Exact rational computation should match manual calculation
        clock = Fraction("1000/180")
        cycles = 2160 + 2 * 10
        expected_ns = math.ceil(cycles * clock)
        
        cost = compute_cluster_service_cost(
            10, "declared_scan_proxy",
            {"clock_ns": "1000/180", "fixed_cycles": 2160, "cycles_per_doc": 2}
        )
        self.assertEqual(cost, expected_ns)
        self.assertEqual(cost, 12112)
    
    def test_reject_bool_cluster_size(self):
        """Reject boolean cluster_size via exact type check."""
        with self.assertRaisesRegex(ValueError, "cluster_size must be non-negative integer"):
            compute_cluster_service_cost(
                True, "declared_scan_proxy",
                {"clock_ns": 1.0, "fixed_cycles": 100, "cycles_per_doc": 2}
            )
    
    def test_reject_bool_parameters(self):
        """Reject boolean for clock_ns, fixed_cycles, cycles_per_doc."""
        for name, params in [
            ("clock_ns", {"clock_ns": True, "fixed_cycles": 100, "cycles_per_doc": 2}),
            ("fixed_cycles", {"clock_ns": 1.0, "fixed_cycles": False, "cycles_per_doc": 2}),
            ("cycles_per_doc", {"clock_ns": 1.0, "fixed_cycles": 100, "cycles_per_doc": True}),
        ]:
            with self.subTest(param=name), self.assertRaisesRegex(ValueError, f"{name} must not be boolean"):
                compute_cluster_service_cost(10, "declared_scan_proxy", params)
    
    def test_reject_nan_inf_parameters(self):
        """Reject NaN and Inf in cost parameters."""
        for name, value in [("NaN", float('nan')), ("Inf", float('inf')), ("-Inf", float('-inf'))]:
            with self.subTest(value=name):
                with self.assertRaisesRegex(ValueError, "must be finite rational"):
                    compute_cluster_service_cost(
                        10, "declared_scan_proxy",
                        {"clock_ns": value, "fixed_cycles": 100, "cycles_per_doc": 2}
                    )
    
    def test_reject_zero_negative_parameters(self):
        """Reject zero and negative cost parameters."""
        test_cases = [
            ("zero_clock", {"clock_ns": 0, "fixed_cycles": 100, "cycles_per_doc": 2}, "clock_ns must be positive"),
            ("negative_clock", {"clock_ns": -1.0, "fixed_cycles": 100, "cycles_per_doc": 2}, "clock_ns must be non-negative"),
            ("zero_fixed", {"clock_ns": 1.0, "fixed_cycles": 0, "cycles_per_doc": 2}, "fixed_cycles must be positive"),
            ("negative_per_doc", {"clock_ns": 1.0, "fixed_cycles": 100, "cycles_per_doc": -2}, "cycles_per_doc must be non-negative"),
        ]
        for name, params, expected_msg in test_cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, expected_msg):
                    compute_cluster_service_cost(10, "declared_scan_proxy", params)
    
    def test_uint64_arrival_overflow(self):
        """Reject arrival_ns exceeding UINT64_MAX."""
        # Create multiple distinct queries to trigger overflow
        base = self.probe_unequal_clusters()
        probes = [
            Probe(base.metadata, f"overflow_{i:04d}", base.centroid_ids,
                  base.cluster_sizes, base.candidate_count)
            for i in range(1000)
        ]
        arrival_spec = {
            "kind": "uniform",
            "rate_qps": "1/1000000000000",  # Very slow rate to cause large offsets
            "start_ns": 2**63,  # Large start value
        }
        model_params = {"clock_ns": 1.0, "fixed_cycles": 100, "cycles_per_doc": 2}
        
        with self.assertRaisesRegex(ValueError, "arrival_ns.*exceeds UINT64_MAX"):
            generate_trace(probes, probes[0].metadata, arrival_spec, "declared_scan_proxy", model_params)
    
    def test_uint64_service_overflow(self):
        """Reject service_ns exceeding UINT64_MAX."""
        with self.assertRaisesRegex(ValueError, "service_ns exceeds UINT64_MAX"):
            compute_cluster_service_cost(
                2**60, "declared_scan_proxy",
                {"clock_ns": 1000.0, "fixed_cycles": 1000, "cycles_per_doc": 1000}
            )
    
    def test_uint64_arrival_plus_service_overflow(self):
        """Reject arrival_ns + service_ns overflow."""
        probe = self.probe_unequal_clusters()
        arrival_spec = {
            "kind": "uniform",
            "rate_qps": 1.0,
            "start_ns": 2**64 - 500,  # Near max
        }
        # Large service cost that would overflow when added to arrival
        model_params = {"clock_ns": 1.0, "fixed_cycles": 10000, "cycles_per_doc": 1}
        
        with self.assertRaisesRegex(ValueError, "arrival_ns \\+ service_ns exceeds UINT64_MAX"):
            generate_trace([probe], probe.metadata, arrival_spec, "declared_scan_proxy", model_params)
    
    def test_per_cluster_cost_not_total_query_cost(self):
        """Each subtask gets its own cluster cost, not total query cost."""
        probe = self.probe_unequal_clusters()
        arrival_spec = {"kind": "uniform", "rate_qps": 1.0, "start_ns": 0}
        model_params = {"clock_ns": "1000/180", "fixed_cycles": 2160, "cycles_per_doc": 2}
        
        tasks = generate_trace([probe], probe.metadata, arrival_spec, "declared_scan_proxy", model_params)
        
        # Two subtasks: one for cluster 0 (size 10), one for cluster 1 (size 100)
        self.assertEqual(len(tasks), 2)
        
        task_0 = [t for t in tasks if t["cluster_id"] == 0][0]
        task_1 = [t for t in tasks if t["cluster_id"] == 1][0]
        
        # Each task has its own cluster-specific cost
        self.assertEqual(task_0["service_ns"], 12112)  # Size 10
        self.assertEqual(task_1["service_ns"], 13112)  # Size 100
        
        # NOT the sum of all cluster costs
        self.assertNotEqual(task_0["service_ns"], 12112 + 13112)
        self.assertNotEqual(task_1["service_ns"], 12112 + 13112)
    
    def test_declared_scan_proxy_label(self):
        """Service cost is labeled 'declared_scan_proxy', not 'calibrated device plan'."""
        # This is tested via manifest in integration tests, but we verify the model name
        probe = self.probe_unequal_clusters()
        arrival_spec = {"kind": "uniform", "rate_qps": 1.0, "start_ns": 0}
        model_params = {"clock_ns": 1.0, "fixed_cycles": 100, "cycles_per_doc": 2}
        
        # Should not raise with correct model name
        tasks = generate_trace([probe], probe.metadata, arrival_spec, "declared_scan_proxy", model_params)
        self.assertGreater(len(tasks), 0)
        
        # Should reject unknown model names
        with self.assertRaisesRegex(ValueError, "Unknown cost_model"):
            generate_trace([probe], probe.metadata, arrival_spec, "calibrated_device_plan", model_params)

    def metadata_for(self, namespace, version):
        return IndexMetadata(
            index_namespace=namespace,
            index_version=version,
            dataset="fiqa",
            index_id="ivfpq8",
            encoder_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            dimension=768,
            metric="IP",
            normalized=True,
            split="test",
            total_centroids=2,
            nprobe=2,
        )

    def test_stable_numeric_query_id_mapping(self):
        """Mapping is deterministic sequential positive uint64 by first occurrence."""
        base = self.probe_unequal_clusters()
        probes = (
            Probe(base.metadata, "query_one", (0, 1), (10, 100), 110),
            Probe(base.metadata, "query_two", (1, 0), (100, 10), 110),
        )
        mapping_a, order_a = build_query_id_mapping(probes, base.metadata)
        mapping_b, order_b = build_query_id_mapping(probes, base.metadata)

        self.assertEqual(mapping_a, mapping_b)
        self.assertEqual(order_a, order_b)
        self.assertEqual([mapping_a[identity] for identity in order_a], [1, 2])
        for query_id in mapping_a.values():
            self.assertIs(type(query_id), int)
            self.assertGreaterEqual(query_id, 1)
            self.assertLessEqual(query_id, UINT64_MAX)

    def test_same_query_subtasks_share_numeric_id(self):
        """Subtasks of one query share its ID; distinct queries do not collide."""
        base = self.probe_unequal_clusters()
        probes = (
            Probe(base.metadata, "query_one", (0, 1), (10, 100), 110),
            Probe(base.metadata, "query_two", (1, 0), (100, 10), 110),
        )
        arrival_spec = {"kind": "uniform", "rate_qps": 1.0, "start_ns": 0}
        model_params = {"clock_ns": 1.0, "fixed_cycles": 100, "cycles_per_doc": 2}
        tasks = generate_trace(probes, base.metadata, arrival_spec,
                              "declared_scan_proxy", model_params)

        self.assertEqual(len(tasks), 4)
        groups = {}
        for task in tasks:
            groups.setdefault(task["probe_query_id"], set()).add(task["query_id"])
        self.assertEqual(groups["query_one"], {1})
        self.assertEqual(groups["query_two"], {2})

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = pathlib.Path(tmpdir) / "trace.csv"
            write_scheduler_csv(tasks, csv_path)
            with csv_path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], [
                "trace", "seed", "task_id", "query_id", "arrival_ns",
                "service_ns", "estimate_ns", "estimate_label", "task_class",
                "query_class",
            ])
            self.assertEqual(len(rows), 5)
            for row in rows[1:]:
                self.assertEqual(len(row), 10)
                self.assertEqual(row[7], "trace_supplied")
                self.assertGreaterEqual(int(row[3]), 1)
                self.assertLessEqual(int(row[3]), UINT64_MAX)
            self.assertNotIn("probe_oracle", csv_path.read_text())

    def test_manifest_preserves_full_identity(self):
        """Manifest keeps namespace, index version, and original query IDs."""
        base = self.probe_unequal_clusters()
        probes = (
            Probe(base.metadata, "query:one,with,delimiters", (0, 1), (10, 100), 110),
            Probe(base.metadata, "query_two", (1, 0), (100, 10), 110),
        )
        arrival_spec = {"kind": "uniform", "rate_qps": 1.0, "start_ns": 0}
        model_params = {"clock_ns": 1.0, "fixed_cycles": 100, "cycles_per_doc": 2}

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = pathlib.Path(tmpdir) / "manifest.json"
            write_manifest(probes, base.metadata, base.cluster_sizes, {},
                          arrival_spec, "declared_scan_proxy", model_params,
                          manifest_path)
            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["csv_estimate_label"], "trace_supplied")
        self.assertEqual(manifest["service_cost_model"]["name"], "declared_scan_proxy")
        self.assertIn("declared scan proxy", manifest["service_cost_model"]["source"])
        mapping = manifest["query_id_mapping"]
        self.assertEqual([entry["query_id"] for entry in mapping], [1, 2])
        self.assertEqual(mapping[0]["index_namespace"], base.metadata.index_namespace)
        self.assertEqual(mapping[0]["index_version"], base.metadata.index_version)
        self.assertEqual(mapping[0]["original_query_id"], "query:one,with,delimiters")
        self.assertEqual(mapping[1]["original_query_id"], "query_two")
        self.assertNotIn("probe_oracle", json.dumps(manifest))

    def test_delimiter_bearing_identity_is_not_joined(self):
        """Structured identity distinguishes values that a delimiter join would merge."""
        metadata_a = self.metadata_for("ns:a", "v:1")
        metadata_b = self.metadata_for("ns", "a:v:1")
        probe_a = Probe(metadata_a, "q", (0, 1), (10, 100), 110)
        probe_b = Probe(metadata_b, "q", (0, 1), (10, 100), 110)

        self.assertNotEqual(query_identity(metadata_a, probe_a),
                            query_identity(metadata_b, probe_b))
        self.assertEqual(":".join(query_identity(metadata_a, probe_a)),
                         ":".join(query_identity(metadata_b, probe_b)))

    def test_duplicate_identity_rejected_before_output(self):
        """Duplicate full identities are rejected rather than collapsed."""
        probe = self.probe_unequal_clusters()
        arrival_spec = {"kind": "uniform", "rate_qps": 1.0, "start_ns": 0}
        model_params = {"clock_ns": 1.0, "fixed_cycles": 100, "cycles_per_doc": 2}

        with self.assertRaisesRegex(ValueError, "duplicate query identity"):
            generate_trace([probe, probe], probe.metadata, arrival_spec,
                           "declared_scan_proxy", model_params)


class FiQASchedulerCLITests(unittest.TestCase):
    """CLI contract tests with synthetic fixtures."""
    
    def setUp(self):
        """Create temporary directory for test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)
        
        # Create synthetic fixtures compatible with probe_contract
        self.metadata = {
            "index_namespace": "test/synthetic",
            "index_version": "v1",
            "dataset": "synthetic",
            "index_id": "test",
            "encoder_model": "sentence-transformers/all-mpnet-base-v2",
            "encoder_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "metric": "IP",
            "normalized": True,
            "split": "test",
            "total_centroids": 2,
            "nprobe": 2,
        }
        
        self.encoder = {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "normalize_embeddings": True,
        }
        
        self.cluster_sizes = [10, 100]
        
        self.probes = [
            {
                "query_id": "q001",
                "query_row": 0,
                "nprobe": 2,
                "centroid_ids": [0, 1],
                "cluster_sizes": [10, 100],
                "candidate_count": 110,
            }
        ]
        
        # Write fixtures to disk
        (self.root / "metadata.json").write_text(json.dumps(self.metadata))
        (self.root / "encoder.json").write_text(json.dumps(self.encoder))
        (self.root / "sizes.json").write_text(json.dumps(self.cluster_sizes))
        (self.root / "probes.jsonl").write_text(json.dumps(self.probes[0]) + "\n")
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.tmpdir.cleanup()
    
    def test_cli_help(self):
        """Test CLI --help output."""
        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fiqa_to_scheduler_trace.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--probes", result.stdout)
        self.assertIn("--clock-ns", result.stdout)
        self.assertIn("--rate-qps", result.stdout)
    
    def test_cli_valid_invocation(self):
        """Test valid CLI invocation with synthetic fixtures."""
        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fiqa_to_scheduler_trace.py"
        output_csv = self.root / "output.csv"
        output_manifest = self.root / "manifest.json"
        
        command = [
            sys.executable, str(script),
            "--probes", str(self.root / "probes.jsonl"),
            "--metadata", str(self.root / "metadata.json"),
            "--cluster-sizes", str(self.root / "sizes.json"),
            "--encoder-config", str(self.root / "encoder.json"),
            "--expected-queries", "1",
            "--output-csv", str(output_csv),
            "--output-manifest", str(output_manifest),
            "--rate-qps", "1.0",
            "--clock-ns", "1000/180",  # Rational string preserved
            "--fixed-cycles", "2160",
            "--cycles-per-doc", "2",
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        
        # Parse JSON output
        output = json.loads(result.stdout)
        self.assertEqual(output["query_count"], 1)
        self.assertEqual(output["subtask_count"], 2)  # nprobe=2
        
        # Verify CSV was created and has correct count
        self.assertTrue(output_csv.exists())
        csv_lines = output_csv.read_text().strip().split("\n")
        self.assertEqual(len(csv_lines), 3)  # Header + 2 tasks
        
        # Verify header
        self.assertIn("trace,seed,task_id,query_id,arrival_ns,service_ns", csv_lines[0])
        
        # Verify task lines contain expected data
        task1_fields = csv_lines[1].split(",")
        task2_fields = csv_lines[2].split(",")
        
        # Both tasks arrive at the same time (fanout model)
        self.assertEqual(task1_fields[4], task2_fields[4])  # arrival_ns
        
        # Different service times for different cluster sizes
        self.assertNotEqual(task1_fields[5], task2_fields[5])  # service_ns
        
        # Verify manifest was created
        self.assertTrue(output_manifest.exists())
        manifest = json.loads(output_manifest.read_text())
        self.assertEqual(manifest["schema"], "cortex.fiqa_scheduler_adapter.v1")
        self.assertEqual(manifest["query_count"], 1)
        self.assertEqual(manifest["subtask_count"], 2)
        self.assertEqual(manifest["service_cost_model"]["name"], "declared_scan_proxy")
        self.assertIn("declared scan proxy", manifest["service_cost_model"]["source"])
        self.assertIn("not complete device service plan", manifest["service_cost_model"]["source"])
        self.assertIn("not calibrated", manifest["service_cost_model"]["source"])


if __name__ == "__main__":
    unittest.main()
