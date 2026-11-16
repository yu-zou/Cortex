#!/usr/bin/env python3
"""Regression tests for fiqa_to_scheduler_trace adapter.

Tests cover negative validation cases and small frozen fixture matching.
"""
import json
import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fiqa_to_scheduler_trace import (
    validate_inputs, compute_cluster_service_cost, generate_trace,
    write_scheduler_csv, write_manifest
)
from scripts.probe_contract import IndexMetadata, Probe


class TestValidateInputs:
    """Test input validation with negative cases."""
    
    def test_reject_missing_probes_file(self, tmp_path):
        """Missing probes file must raise FileNotFoundError."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        # Create other files but not probes
        metadata.write_text('{}')
        sizes.write_text('[]')
        encoder.write_text('{}')
        
        with pytest.raises(FileNotFoundError, match="Probes file not found"):
            validate_inputs(probes, metadata, sizes, encoder, 1)
    
    def test_reject_missing_metadata_file(self, tmp_path):
        """Missing metadata file must raise FileNotFoundError."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        probes.write_text('')
        sizes.write_text('[]')
        encoder.write_text('{}')
        
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            validate_inputs(probes, metadata, sizes, encoder, 1)
    
    def test_reject_missing_cluster_sizes_file(self, tmp_path):
        """Missing cluster sizes file must raise FileNotFoundError."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        probes.write_text('')
        metadata.write_text('{}')
        encoder.write_text('{}')
        
        with pytest.raises(FileNotFoundError, match="Cluster sizes file not found"):
            validate_inputs(probes, metadata, sizes, encoder, 1)
    
    def test_reject_missing_encoder_config_file(self, tmp_path):
        """Missing encoder config file must raise FileNotFoundError."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        probes.write_text('')
        metadata.write_text('{}')
        sizes.write_text('[]')
        
        with pytest.raises(FileNotFoundError, match="Encoder config file not found"):
            validate_inputs(probes, metadata, sizes, encoder, 1)
    
    def test_reject_zero_expected_queries(self, tmp_path):
        """Zero expected_queries must raise ValueError."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        for f in [probes, metadata, sizes, encoder]:
            f.write_text('{}')
        
        with pytest.raises(ValueError, match="expected_queries must be positive"):
            validate_inputs(probes, metadata, sizes, encoder, 0)
    
    def test_reject_negative_expected_queries(self, tmp_path):
        """Negative expected_queries must raise ValueError."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        for f in [probes, metadata, sizes, encoder]:
            f.write_text('{}')
        
        with pytest.raises(ValueError, match="expected_queries must be positive"):
            validate_inputs(probes, metadata, sizes, encoder, -1)
    
    def test_reject_wrong_dimension(self, tmp_path):
        """Dimension mismatch between metadata and encoder must be rejected."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        metadata_data = {
            "index_namespace": "test",
            "index_version": "v1",
            "dataset": "test",
            "index_id": "test_id",
            "encoder_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "metric": "IP",
            "normalized": True,
            "split": "test",
            "total_centroids": 2,
            "nprobe": 1,
            "encoder_model": "sentence-transformers/all-mpnet-base-v2"
        }
        encoder_data = {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 512,  # Wrong dimension
            "normalize_embeddings": True
        }
        
        metadata.write_text(json.dumps(metadata_data))
        encoder.write_text(json.dumps(encoder_data))
        sizes.write_text(json.dumps([10, 20]))
        probes.write_text('')
        
        with pytest.raises(ValueError, match="encoder config mismatch"):
            validate_inputs(probes, metadata, sizes, encoder, 1)
    
    def test_reject_cluster_size_cardinality_mismatch(self, tmp_path):
        """Cluster sizes length must match total_centroids."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        metadata_data = {
            "index_namespace": "test",
            "index_version": "v1",
            "dataset": "test",
            "index_id": "test_id",
            "encoder_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "metric": "IP",
            "normalized": True,
            "split": "test",
            "total_centroids": 3,  # Expects 3 sizes
            "nprobe": 1,
            "encoder_model": "sentence-transformers/all-mpnet-base-v2"
        }
        encoder_data = {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "normalize_embeddings": True
        }
        
        metadata.write_text(json.dumps(metadata_data))
        encoder.write_text(json.dumps(encoder_data))
        sizes.write_text(json.dumps([10, 20]))  # Only 2 sizes
        probes.write_text('')
        
        with pytest.raises(ValueError, match="cluster_sizes length .* != total_centroids"):
            validate_inputs(probes, metadata, sizes, encoder, 1)
    
    def test_reject_query_count_mismatch(self, tmp_path):
        """Actual query count must match expected_queries."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        metadata_data = {
            "index_namespace": "test",
            "index_version": "v1",
            "dataset": "test",
            "index_id": "test_id",
            "encoder_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "metric": "IP",
            "normalized": True,
            "split": "test",
            "total_centroids": 2,
            "nprobe": 1,
            "encoder_model": "sentence-transformers/all-mpnet-base-v2"
        }
        encoder_data = {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "normalize_embeddings": True
        }
        probe_line = json.dumps({
            "query_id": "q1",
            "centroid_ids": [0],
            "cluster_sizes": [100],
            "candidate_count": 100
        })
        
        metadata.write_text(json.dumps(metadata_data))
        encoder.write_text(json.dumps(encoder_data))
        sizes.write_text(json.dumps([100, 200]))
        probes.write_text(probe_line + '\n')
        
        with pytest.raises(ValueError, match="Found 1 probes but expected 2"):
            validate_inputs(probes, metadata, sizes, encoder, 2)
    
    def test_reject_duplicate_query_ids(self, tmp_path):
        """Duplicate query IDs must be rejected."""
        probes = tmp_path / "probes.jsonl"
        metadata = tmp_path / "metadata.json"
        sizes = tmp_path / "sizes.json"
        encoder = tmp_path / "encoder.json"
        
        metadata_data = {
            "index_namespace": "test",
            "index_version": "v1",
            "dataset": "test",
            "index_id": "test_id",
            "encoder_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "metric": "IP",
            "normalized": True,
            "split": "test",
            "total_centroids": 2,
            "nprobe": 1,
            "encoder_model": "sentence-transformers/all-mpnet-base-v2"
        }
        encoder_data = {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "normalize_embeddings": True
        }
        probe_line = json.dumps({
            "query_id": "q1",
            "centroid_ids": [0],
            "cluster_sizes": [100],
            "candidate_count": 100
        })
        
        metadata.write_text(json.dumps(metadata_data))
        encoder.write_text(json.dumps(encoder_data))
        sizes.write_text(json.dumps([100, 200]))
        probes.write_text(probe_line + '\n' + probe_line + '\n')  # Duplicate
        
        with pytest.raises(ValueError, match="duplicate query"):
            validate_inputs(probes, metadata, sizes, encoder, 2)


class TestComputeClusterServiceCost:
    """Test per-cluster service cost computation."""
    
    def test_declared_scan_proxy_cost(self):
        """Test declared scan proxy cost model with exact arithmetic."""
        model_params = {
            "clock_ns": 1000 / 180,  # 180 MHz = 5.555... ns
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        # cluster_size = 1000
        cost_ns = compute_cluster_service_cost(1000, "declared_scan_proxy", model_params)
        
        # Expected: ceil((2160 + 2 * 1000) * (1000/180)) = ceil(23111.111...) = 23112 ns
        import math
        from fractions import Fraction
        expected = math.ceil((Fraction(2160) + Fraction(2) * 1000) * Fraction(1000, 180))
        assert cost_ns == expected
        assert cost_ns == 23112
    
    def test_unequal_clusters_different_costs(self):
        """Unequal cluster sizes must produce different service costs."""
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        cost_10 = compute_cluster_service_cost(10, "declared_scan_proxy", model_params)
        cost_100 = compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
        
        # 10 docs: ceil((2160 + 2*10) * (1000/180)) = ceil(12111.111...) = 12112
        # 100 docs: ceil((2160 + 2*100) * (1000/180)) = ceil(13111.111...) = 13112
        import math
        from fractions import Fraction
        expected_10 = math.ceil((Fraction(2160) + Fraction(2) * 10) * Fraction(1000, 180))
        expected_100 = math.ceil((Fraction(2160) + Fraction(2) * 100) * Fraction(1000, 180))
        
        assert cost_10 == expected_10
        assert cost_100 == expected_100
        assert cost_10 != cost_100
        assert cost_100 > cost_10
    
    def test_fractional_ceil_case(self):
        """Fractional nanoseconds must ceil to integer."""
        model_params = {
            "clock_ns": 1.5,  # Fractional clock period
            "fixed_cycles": 10.0,
            "cycles_per_doc": 0.5,
        }
        
        # cluster_size = 3: (10 + 0.5*3) * 1.5 = 11.5 * 1.5 = 17.25 -> ceil = 18
        cost_ns = compute_cluster_service_cost(3, "declared_scan_proxy", model_params)
        assert cost_ns == 18
    
    def test_reject_missing_clock_ns(self):
        """Missing clock_ns must be rejected."""
        model_params = {
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="declared_scan_proxy requires"):
            compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
    
    def test_reject_missing_fixed_cycles(self):
        """Missing fixed_cycles must be rejected."""
        model_params = {
            "clock_ns": 1000 / 180,
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="declared_scan_proxy requires"):
            compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
    
    def test_reject_missing_cycles_per_doc(self):
        """Missing cycles_per_doc must be rejected."""
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
        }
        
        with pytest.raises(ValueError, match="declared_scan_proxy requires"):
            compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
    
    def test_reject_bool_as_number(self):
        """Boolean values must be rejected."""
        model_params = {
            "clock_ns": True,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="must not be boolean"):
            compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
    
    def test_reject_nan_clock(self):
        """NaN clock_ns must be rejected."""
        model_params = {
            "clock_ns": float('nan'),
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="must be finite rational"):
            compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
    
    def test_reject_inf_cycles(self):
        """Infinite cycles must be rejected."""
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": float('inf'),
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="must be finite rational"):
            compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
    
    def test_reject_zero_clock_ns(self):
        """Zero clock_ns must be rejected."""
        model_params = {
            "clock_ns": 0.0,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="must be positive"):
            compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
    
    def test_reject_negative_cycles_per_doc(self):
        """Negative cycles_per_doc must be rejected."""
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": -2.0,
        }
        
        with pytest.raises(ValueError, match="must be positive"):
            compute_cluster_service_cost(100, "declared_scan_proxy", model_params)
    
    def test_reject_negative_cluster_size(self):
        """Negative cluster size must be rejected."""
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="cluster_size must be non-negative integer"):
            compute_cluster_service_cost(-1, "declared_scan_proxy", model_params)
    
    def test_reject_uint64_overflow(self):
        """Service cost exceeding UINT64_MAX must be rejected."""
        model_params = {
            "clock_ns": 1e15,  # Very large clock period
            "fixed_cycles": 1e10,
            "cycles_per_doc": 1.0,
        }
        
        with pytest.raises(ValueError, match="exceeds UINT64_MAX"):
            compute_cluster_service_cost(1000, "declared_scan_proxy", model_params)
    
    def test_reject_unknown_cost_model(self):
        """Unknown cost model must be rejected."""
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="Unknown cost_model"):
            compute_cluster_service_cost(100, "unknown_model", model_params)


class TestGenerateTrace:
    """Test trace generation."""
    
    def test_reject_non_uniform_arrival(self):
        """Non-uniform arrival must be rejected."""
        arrival_spec = {"kind": "replay"}
        
        with pytest.raises(ValueError, match="Only uniform arrival replay supported"):
            generate_trace([], None, arrival_spec, "declared_scan_proxy", {})
    
    def test_reject_bool_rate_qps(self):
        """Boolean rate_qps must be rejected."""
        arrival_spec = {"kind": "uniform", "rate_qps": True}
        
        with pytest.raises(ValueError, match="rate_qps must not be boolean"):
            generate_trace([], None, arrival_spec, "declared_scan_proxy", {})
    
    def test_reject_nan_rate_qps(self):
        """NaN rate_qps must be rejected."""
        arrival_spec = {"kind": "uniform", "rate_qps": float('nan')}
        
        with pytest.raises(ValueError, match="must be finite rational"):
            generate_trace([], None, arrival_spec, "declared_scan_proxy", {})
    
    def test_reject_negative_rate_qps(self):
        """Negative rate_qps must be rejected."""
        arrival_spec = {"kind": "uniform", "rate_qps": -1.0}
        
        with pytest.raises(ValueError, match="rate_qps must be positive"):
            generate_trace([], None, arrival_spec, "declared_scan_proxy", {})
    
    def test_reject_zero_rate_qps(self):
        """Zero rate_qps must be rejected."""
        arrival_spec = {"kind": "uniform", "rate_qps": 0.0}
        
        with pytest.raises(ValueError, match="rate_qps must be positive"):
            generate_trace([], None, arrival_spec, "declared_scan_proxy", {})
    
    def test_reject_negative_start_ns(self):
        """Negative start_ns must be rejected."""
        arrival_spec = {"kind": "uniform", "rate_qps": 10.0, "start_ns": -100}
        
        with pytest.raises(ValueError, match="start_ns must be non-negative"):
            generate_trace([], None, arrival_spec, "declared_scan_proxy", {})
    
    def test_generate_correct_subtask_count(self):
        """Each probe must generate nprobe subtasks."""
        metadata = IndexMetadata(
            index_namespace="test",
            index_version="v1",
            dataset="test",
            index_id="test_id",
            encoder_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            dimension=768,
            metric="IP",
            normalized=True,
            split="test",
            total_centroids=3,
            nprobe=2,
            encoder_model="sentence-transformers/all-mpnet-base-v2"
        )
        
        probes = [
            Probe(metadata, "q1", (0, 1), (100, 200), 300),
            Probe(metadata, "q2", (1, 2), (200, 300), 500),
        ]
        
        arrival_spec = {"kind": "uniform", "rate_qps": 10.0}
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        tasks = generate_trace(probes, metadata, arrival_spec, "declared_scan_proxy", model_params)
        
        # 2 queries * 2 nprobe = 4 subtasks
        assert len(tasks) == 4
    
    def test_per_cluster_cost_not_repeated_sum(self):
        """Service cost must be per-cluster, not sum of all candidates."""
        metadata = IndexMetadata(
            index_namespace="test",
            index_version="v1",
            dataset="test",
            index_id="test_id",
            encoder_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            dimension=768,
            metric="IP",
            normalized=True,
            split="test",
            total_centroids=3,
            nprobe=2,
            encoder_model="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Probe with unequal clusters: 10 and 100 docs
        probes = [Probe(metadata, "q1", (0, 1), (10, 100), 110)]
        
        arrival_spec = {"kind": "uniform", "rate_qps": 10.0}
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        tasks = generate_trace(probes, metadata, arrival_spec, "declared_scan_proxy", model_params)
        
        # Should have 2 subtasks with DIFFERENT costs
        assert len(tasks) == 2
        assert tasks[0]["cluster_size"] == 10
        assert tasks[1]["cluster_size"] == 100
        
        # Costs must be different (not both using candidate_count=110)
        assert tasks[0]["service_ns"] != tasks[1]["service_ns"]
        
        # Cost should be based on individual cluster sizes
        import math
        from fractions import Fraction
        expected_cost_10 = math.ceil((Fraction(2160) + Fraction(2) * 10) * Fraction(1000, 180))
        expected_cost_100 = math.ceil((Fraction(2160) + Fraction(2) * 100) * Fraction(1000, 180))
        
        assert tasks[0]["service_ns"] == expected_cost_10
        assert tasks[1]["service_ns"] == expected_cost_100
    
    def test_uniform_arrival_timing(self):
        """Uniform arrivals must use floor of rational schedule."""
        metadata = IndexMetadata(
            index_namespace="test",
            index_version="v1",
            dataset="test",
            index_id="test_id",
            encoder_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            dimension=768,
            metric="IP",
            normalized=True,
            split="test",
            total_centroids=2,
            nprobe=1,
            encoder_model="sentence-transformers/all-mpnet-base-v2"
        )
        
        probes = [
            Probe(metadata, "q1", (0,), (100,), 100),
            Probe(metadata, "q2", (1,), (200,), 200),
            Probe(metadata, "q3", (0,), (150,), 150),
        ]
        
        arrival_spec = {"kind": "uniform", "rate_qps": 3.0, "start_ns": 1000}
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        tasks = generate_trace(probes, metadata, arrival_spec, "declared_scan_proxy", model_params)
        
        # Expected arrivals at 3 QPS starting from 1000 ns:
        # Query 0: 1000 + 0 * 1e9 / 3 = 1000
        # Query 1: 1000 + 1 * 1e9 / 3 = 333333334
        # Query 2: 1000 + 2 * 1e9 / 3 = 666666667
        assert tasks[0]["arrival_ns"] == 1000
        assert tasks[1]["arrival_ns"] == 333333334
        assert tasks[2]["arrival_ns"] == 666666667
    
    def test_arrival_plus_service_overflow(self):
        """arrival_ns + service_ns exceeding UINT64_MAX must be rejected."""
        metadata = IndexMetadata(
            index_namespace="test",
            index_version="v1",
            dataset="test",
            index_id="test_id",
            encoder_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            dimension=768,
            metric="IP",
            normalized=True,
            split="test",
            total_centroids=1,
            nprobe=1,
            encoder_model="sentence-transformers/all-mpnet-base-v2"
        )
        
        probes = [Probe(metadata, "q1", (0,), (100,), 100)]
        
        # Very late arrival near UINT64_MAX
        arrival_spec = {"kind": "uniform", "rate_qps": 1e-15, "start_ns": 2**64 - 1000}
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        with pytest.raises(ValueError, match="exceeds UINT64_MAX"):
            generate_trace(probes, metadata, arrival_spec, "declared_scan_proxy", model_params)


class TestWriteSchedulerCSV:
    """Test CSV output format."""
    
    def test_csv_header(self, tmp_path):
        """CSV must have correct header."""
        output = tmp_path / "trace.csv"
        tasks = []
        
        write_scheduler_csv(tasks, output)
        
        lines = output.read_text().splitlines()
        assert lines[0] == ("trace,seed,task_id,query_id,arrival_ns,service_ns,estimate_ns,"
                           "estimate_label,task_class,query_class")
    
    def test_csv_format(self, tmp_path):
        """CSV rows must match scheduler_trace_runner format."""
        output = tmp_path / "trace.csv"
        tasks = [{
            "task_id": 1,
            "query_id": 42,
            "arrival_ns": 1000,
            "service_ns": 50000,
            "estimate_ns": 50000,
            "task_class": "cluster_0",
            "query_class": "nprobe2_cands300",
        }]
        
        write_scheduler_csv(tasks, output)
        
        lines = output.read_text().splitlines()
        assert len(lines) == 2  # Header + 1 task
        assert lines[1] == "fiqa_real,0,1,42,1000,50000,50000,trace_supplied,cluster_0,nprobe2_cands300"


class TestEndToEnd:
    """End-to-end tests with small frozen fixture."""
    
    @pytest.fixture
    def small_fixture(self, tmp_path):
        """Create a minimal valid fixture."""
        metadata_data = {
            "index_namespace": "fiqa",
            "index_version": "v1_test",
            "dataset": "fiqa",
            "index_id": "test_ivfpq",
            "encoder_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "metric": "IP",
            "normalized": True,
            "split": "test",
            "total_centroids": 3,
            "nprobe": 2,
            "encoder_model": "sentence-transformers/all-mpnet-base-v2"
        }
        encoder_data = {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "dimension": 768,
            "normalize_embeddings": True
        }
        cluster_sizes = [100, 200, 150]
        probes = [
            {"query_id": "FIQA_Q1", "centroid_ids": [0, 1], "cluster_sizes": [100, 200], "candidate_count": 300},
            {"query_id": "FIQA_Q2", "centroid_ids": [1, 2], "cluster_sizes": [200, 150], "candidate_count": 350},
        ]
        
        metadata_path = tmp_path / "metadata.json"
        encoder_path = tmp_path / "encoder.json"
        sizes_path = tmp_path / "cluster_sizes.json"
        probes_path = tmp_path / "probes.jsonl"
        
        metadata_path.write_text(json.dumps(metadata_data))
        encoder_path.write_text(json.dumps(encoder_data))
        sizes_path.write_text(json.dumps(cluster_sizes))
        probes_path.write_text('\n'.join(json.dumps(p) for p in probes) + '\n')
        
        return {
            "metadata": metadata_path,
            "encoder": encoder_path,
            "sizes": sizes_path,
            "probes": probes_path,
            "expected_queries": 2,
        }
    
    def test_complete_flow(self, small_fixture, tmp_path):
        """Test complete adapter flow with small fixture."""
        from scripts.fiqa_to_scheduler_trace import validate_inputs, generate_trace, write_scheduler_csv, write_manifest
        
        # Validate inputs
        probes, metadata, cluster_sizes = validate_inputs(
            small_fixture["probes"],
            small_fixture["metadata"],
            small_fixture["sizes"],
            small_fixture["encoder"],
            small_fixture["expected_queries"]
        )
        
        # Check validated data
        assert len(probes) == 2
        assert metadata.nprobe == 2
        assert len(cluster_sizes) == 3
        
        # Generate trace
        arrival_spec = {"kind": "uniform", "rate_qps": 10.0, "start_ns": 0}
        model_params = {
            "clock_ns": 1000 / 180,
            "fixed_cycles": 2160.0,
            "cycles_per_doc": 2.0,
        }
        
        tasks = generate_trace(probes, metadata, arrival_spec, "declared_scan_proxy", model_params)
        
        # Check task count: 2 queries * 2 nprobe = 4 subtasks
        assert len(tasks) == 4
        
        # Write outputs
        csv_path = tmp_path / "trace.csv"
        manifest_path = tmp_path / "manifest.json"
        
        write_scheduler_csv(tasks, csv_path)
        write_manifest(probes, metadata, cluster_sizes,
                      {"probes": small_fixture["probes"],
                       "metadata": small_fixture["metadata"],
                       "sizes": small_fixture["sizes"],
                       "encoder": small_fixture["encoder"]},
                      arrival_spec, "declared_scan_proxy", model_params, manifest_path)
        
        # Verify CSV
        csv_lines = csv_path.read_text().splitlines()
        assert len(csv_lines) == 5  # Header + 4 tasks
        assert "fiqa_real" in csv_lines[1]
        
        # Verify manifest
        manifest = json.loads(manifest_path.read_text())
        assert manifest["query_count"] == 2
        assert manifest["subtask_count"] == 4
        assert manifest["metadata"]["dimension"] == 768
        assert manifest["service_cost_model"]["name"] == "declared_scan_proxy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
