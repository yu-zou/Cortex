"""Tiny synthetic contract fixtures; never performance or cloud-price evidence."""
import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import comparison_protocol as cp


def fixture():
    h = "a" * 64
    protocol = dict(
        dataset="synthetic", corpus_sha256=h, queryset_sha256="b" * 64,
        qrels_sha256="c" * 64, train_queryset_sha256="d" * 64,
        validation_queryset_sha256="e" * 64, split="test", tuning_split="validation",
        embedding=dict(model="synthetic-encoder", revision="f" * 40,
                       dimension=4, normalized=True),
        metric="IP", direction="higher", top_k=2,
        rerank=dict(method="exact_IP", candidates=4, scope="per_query_union"),
        filter="none", failure_policy="count_all_failures", cache_mode="warm",
        index_namespace="synthetic/tenant-a", index_version="v1",
        quality=dict(metric="ndcg", target="qrels", k=2, threshold=0.7,
                     tolerance=0.02, registration_sha256=h,
                     declared_at="before_validation"))
    host = dict(instance="fixture-host", instance_type="synthetic", cpu_vcpu=4,
                memory_bytes=4096, network_bytes_per_s=1000, devices=[],
                processes=[dict(role="all-services-including-control-plane",
                                cpu_vcpu=4, memory_bytes=4096)])
    resources = dict(instances=[host], client=dict(cpu_vcpu=1, memory_bytes=1024,
                     network_bytes_per_s=1000, instance_type="synthetic-client"),
                     storage=dict(storage_class="synthetic-block", disks=1,
                                  capacity_bytes=8192, iops=100,
                                  bandwidth_bytes_per_s=1000, replicas=1,
                                  min_size=1, failure_domain="host", io_path="block",
                                  physical_pool_sha256=h))
    config = dict(nprobe=2, R=4)
    return dict(schema_version=1, category="same_resource", protocol=protocol,
                index=dict(sha256=h, namespace="synthetic/tenant-a", version="v1"),
                selection=dict(split="validation", config=config,
                               config_sha256=cp.digest(config), evidence_sha256=h,
                               validation_score=0.73, final_test_score=0.71,
                               test_used_for_selection=False),
                backend=dict(kind="cpu", real_components=["cpu", "storage", "network"],
                             modeled_components=[], result_source="online_compute",
                             evidence_sha256=h, profile_sha256=None,
                             device_sha256=None, bit_sha256=None),
                resources=resources,
                cost=dict(currency="USD", unit="currency_total", build=0, run=0,
                          storage=0, client=0, duration_s=10,
                          evidence_sha256=h, basis="unit_fixture"))


class ComparisonProtocolTest(unittest.TestCase):
    def reject(self, change, message):
        run = fixture()
        change(run)
        with self.assertRaisesRegex(ValueError, message):
            cp.validate_run(run)

    def test_positive_and_input_not_mutated(self):
        run = fixture()
        before = copy.deepcopy(run)
        result = cp.compare_runs(run, copy.deepcopy(run))
        self.assertTrue(result["quality_matched"])
        self.assertEqual(result["claim"], "software_comparison")
        self.assertEqual(run, before)

    def test_invalid_metric_and_direction(self):
        for key, value in (("metric", "cosine-maybe"), ("direction", "lower")):
            self.reject(lambda r: r["protocol"].update({key: value}), "metric|direction")

    def test_test_tuning_rejected(self):
        self.reject(lambda r: r["selection"].update(split="test"), "validation")
        self.reject(lambda r: r["selection"].update(test_used_for_selection=True), "test")
        self.reject(lambda r: r["protocol"].update(tuning_split="test"), "validation")

    def test_resource_mismatch_even_one_unit(self):
        a, b = fixture(), fixture()
        b["resources"]["instances"][0]["memory_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "resource"):
            cp.compare_runs(a, b)

    def test_ceph_milvus_roles_differ_on_reused_hardware(self):
        a, b = fixture(), fixture()
        a["resources"]["instances"][0]["processes"] = [
            dict(role="MON/MGR/RGW", cpu_vcpu=1, memory_bytes=1024),
            dict(role="OSD", cpu_vcpu=3, memory_bytes=3072)]
        b["resources"]["instances"][0]["processes"] = [
            dict(role="Milvus", cpu_vcpu=2, memory_bytes=2048),
            dict(role="etcd", cpu_vcpu=1, memory_bytes=1024)]
        b["index"]["sha256"] = "9" * 64
        before = copy.deepcopy((a, b))
        result = cp.compare_runs(a, b)
        self.assertTrue(result["equal_resource_claim"])
        self.assertEqual((a, b), before)

    def test_reused_pool_instance_ids_remain_exact(self):
        a, b = fixture(), fixture()
        b["resources"]["instances"][0]["instance"] = "replacement-host"
        with self.assertRaisesRegex(ValueError, "resource"):
            cp.compare_runs(a, b)

    def test_resource_inventory_order_is_not_hardware(self):
        a, b = fixture(), fixture()
        host = copy.deepcopy(a["resources"]["instances"][0])
        host["instance"] = "second-host"
        for run in (a, b):
            run["resources"]["instances"].append(copy.deepcopy(host))
        b["resources"]["instances"].reverse()
        self.assertTrue(cp.compare_runs(a, b)["equal_resource_claim"])

    def test_different_io_paths_disclosed_conditionally(self):
        a, b = fixture(), fixture()
        a["resources"]["storage"]["io_path"] = "Ceph/BlueStore"
        b["resources"]["storage"]["io_path"] = "Milvus/filesystem"
        result = cp.compare_runs(a, b)
        self.assertTrue(result["equal_resource_claim"])
        self.assertTrue(result["architecture_difference"])
        self.assertEqual(result["io_paths"], ["Ceph/BlueStore", "Milvus/filesystem"])
        self.assertEqual(result["claim"], "conditional_architecture_comparison")

    def test_unestablished_physical_storage_requires_architecture_specific(self):
        for evidence in (None, "9" * 64):
            a, b = fixture(), fixture()
            b["resources"]["storage"]["physical_pool_sha256"] = evidence
            with self.assertRaisesRegex(ValueError, "architecture_specific"):
                cp.compare_runs(a, b)
            a["category"] = b["category"] = "architecture_specific"
            self.assertFalse(cp.compare_runs(a, b)["equal_resource_claim"])

    def test_all_provisioned_quotas_remain_exact(self):
        for section, key in (("storage", "iops"), ("storage", "replicas"),
                             ("client", "cpu_vcpu"), ("client", "network_bytes_per_s")):
            a, b = fixture(), fixture()
            b["resources"][section][key] += 1
            with self.assertRaisesRegex(ValueError, "resource"):
                cp.compare_runs(a, b)

    def test_offline_sources_cannot_enter_online_gate(self):
        self.reject(lambda r: r["backend"].update(kind="offline_des"), "offline")
        for source in ("offline_projection", "response_oracle"):
            self.reject(lambda r: r["backend"].update(result_source=source), "online")

    def test_test_pair_must_match_even_above_threshold(self):
        a, b = fixture(), fixture()
        b["selection"]["final_test_score"] = 0.9
        self.assertFalse(cp.compare_runs(a, b)["quality_matched"])

    def test_validation_threshold_not_relaxed_by_test_tolerance(self):
        a, b = fixture(), fixture()
        b["selection"]["validation_score"] = 0.69
        self.assertFalse(cp.compare_runs(a, b)["quality_matched"])

    def test_protocol_mismatch(self):
        for key, value in (("queryset_sha256", "9" * 64), ("top_k", 1)):
            a, b = fixture(), fixture()
            b["protocol"][key] = value
            if key == "top_k":
                b["protocol"]["quality"]["k"] = value
            with self.assertRaisesRegex(ValueError, "protocol"):
                cp.compare_runs(a, b)

    def test_namespace_and_version(self):
        self.reject(lambda r: r["index"].update(namespace="other"), "namespace")
        self.reject(lambda r: r["index"].update(version="v2"), "namespace")
        for protocol_key, index_key in (("index_namespace", "namespace"),
                                        ("index_version", "version")):
            a, b = fixture(), fixture()
            b["protocol"][protocol_key] = b["index"][index_key] = "other"
            with self.assertRaisesRegex(ValueError, "protocol"):
                cp.compare_runs(a, b)

    def test_secret_bearing_config(self):
        for secret in ({"api_key": "never-print-me"}, {"nested": {"password": "x"}},
                       {"url": "https://user:password@example.invalid"},
                       {"option": "Bearer abcdef"}):
            def change(r):
                r["selection"]["config"] = secret
                r["selection"]["config_sha256"] = cp.digest(secret)
            self.reject(change, "secret|credential")

    def test_config_hash_binding(self):
        self.reject(lambda r: r["selection"]["config"].update(R=8), "config.*hash")

    def test_quality_failure_not_fabricated(self):
        a, b = fixture(), fixture()
        b["selection"]["final_test_score"] = 0.6
        result = cp.compare_runs(a, b)
        self.assertFalse(result["quality_matched"])
        self.assertEqual(result["claim"], "unmatched_quality")

    def test_final_test_does_not_change_selection(self):
        run = fixture()
        run["selection"]["validation_score"] = 0.5
        run["selection"]["final_test_score"] = 1.0
        self.assertFalse(cp.compare_runs(run, fixture())["quality_matched"])

    def test_predeclared_gate_and_metric_targets(self):
        self.reject(lambda r: r["protocol"]["quality"].update(declared_at="after_test"),
                    "predeclared")
        self.reject(lambda r: r["protocol"]["quality"].update(target="ann_overlap"),
                    "quality")
        self.reject(lambda r: r["protocol"].update(validation_queryset_sha256="b" * 64),
                    "split")

    def test_l2_and_ann_overlap_are_explicit(self):
        run = fixture()
        run["protocol"].update(metric="L2", direction="lower")
        run["protocol"]["rerank"]["method"] = "exact_L2"
        run["protocol"]["quality"].update(metric="overlap", target="ann_exact")
        cp.validate_run(run)

    def test_malformed_component_fails_closed(self):
        self.reject(lambda r: r["backend"].update(real_components=[{}]), "text")

    def test_emulation_is_conditional(self):
        a, b = fixture(), fixture()
        b["backend"].update(kind="emulated", modeled_components=["fpga"],
                             result_source="response_oracle", profile_sha256="1" * 64)
        self.assertEqual(cp.compare_runs(a, b)["claim"], "conditional_emulation_comparison")

    def test_physical_requires_evidence_and_online_results(self):
        self.reject(lambda r: r["backend"].update(kind="physical"), "physical")
        run = fixture()
        run["backend"].update(kind="physical", device_sha256="1" * 64,
                              bit_sha256="2" * 64)
        self.assertEqual(cp.compare_runs(fixture(), run)["claim"],
                         "physical_backend_comparison")
        run["backend"]["result_source"] = "response_oracle"
        with self.assertRaises(ValueError):
            cp.validate_run(run)

    def test_same_cost_exact_and_units(self):
        a, b = fixture(), fixture()
        a["category"] = b["category"] = "same_cost"
        b["resources"]["instances"][0]["memory_bytes"] += 1
        self.assertTrue(cp.compare_runs(a, b)["quality_matched"])
        b["cost"]["run"] = 1
        with self.assertRaisesRegex(ValueError, "cost"):
            cp.compare_runs(a, b)
        self.reject(lambda r: r["cost"].update(currency="dollars"), "currency")
        self.reject(lambda r: r["cost"].update(unit="per_hour"), "unit")

    def test_architecture_specific_not_equal_resource(self):
        a, b = fixture(), fixture()
        a["category"] = b["category"] = "architecture_specific"
        b["resources"]["storage"]["io_path"] = "object"
        self.assertFalse(cp.compare_runs(a, b)["equal_resource_claim"])

    def test_numeric_unknown_fields_and_quota_validation(self):
        self.reject(lambda r: r["cost"].update(run=float("nan")), "finite")
        self.reject(lambda r: r["protocol"].update(top_k=True), "integer")
        self.reject(lambda r: r.update(unrecognized="x"), "fields")
        self.reject(lambda r: r["resources"]["instances"][0]["processes"][0].update(
            cpu_vcpu=5), "quota")


if __name__ == "__main__":
    unittest.main()
