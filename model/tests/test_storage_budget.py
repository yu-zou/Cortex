"""Fail-closed budgeting and historical offline-pilot comparison only.

Run with:  PYTHONPATH=scripts .venv/bin/python -m pytest -q tests/test_storage_budget.py
"""
import pathlib
import json
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import storage_budget as sb  # noqa: E402

# Measured on the workstation 2026-11-01 via `du -sh`:
#   ~/strata_sim/data/benchmarks/fiqa     = 407 MiB
#   ~/strata_sim/data/benchmarks/scifact  =  61 MiB  -> 468 MiB dataset dirs
#   ~/strata_sim/data/benchmarks          = 905 MiB (incl. one model copy)
PILOT_DATASETS_ACTUAL = 468 * 1024 ** 2
PILOT_TOTAL_ACTUAL = 905 * 1024 ** 2
FREE_BYTES = 71_802_556_416  # df -B1 / on 2026-11-01


class StorageBudgetTest(unittest.TestCase):
    def test_budget_zero_refuses(self):
        plan = sb.estimate_plan(
            ["scifact", "fiqa"], free_bytes=FREE_BYTES, budget_bytes=0
        )
        self.assertTrue(plan["refusals"])
        self.assertIn("budget is zero", " ".join(plan["refusals"]))
        self.assertFalse(plan["fits_budget"])

    def test_insufficient_free_refuses(self):
        plan = sb.estimate_plan(
            ["hotpotqa"], free_bytes=1_000_000_000, budget_bytes=FREE_BYTES
        )
        self.assertTrue(plan["refusals"])
        self.assertIn("exceeds available free space", " ".join(plan["refusals"]))
        self.assertFalse(plan["arithmetic_fits_free"])
        self.assertFalse(plan["download_ready"])

    def test_both_large_refused_without_approval(self):
        plan = sb.estimate_plan(
            ["hotpotqa", "nq"], free_bytes=FREE_BYTES, budget_bytes=FREE_BYTES
        )
        self.assertTrue(plan["refusals"])
        self.assertIn("both large corpora", " ".join(plan["refusals"]))

        approved = sb.estimate_plan(
            ["hotpotqa", "nq"],
            free_bytes=FREE_BYTES,
            budget_bytes=FREE_BYTES,
            allow_both_large=True,
        )
        self.assertTrue(approved["refusals"])
        self.assertFalse(approved["download_ready"])

    def test_unknown_large_corpus_refuses_even_with_ample_space(self):
        for name in ("hotpotqa", "nq"):
            plan = sb.estimate_plan(
                [name], free_bytes=FREE_BYTES, budget_bytes=FREE_BYTES
            )
            self.assertTrue(plan["refusals"], name)
            self.assertFalse(plan["download_ready"], name)
            self.assertEqual(plan["estimate_status"], "lower_bound")
            self.assertIsNone(plan["fits_free"])

    def test_invalid_dataset_numbers(self):
        for field, values in {
            "docs": [-1, 1.5, float("nan")], "queries": [-1],
            "corpus_bytes": [-1, float("nan"), 1.2], "dim": [0, -1, 767, float("nan"), True],
            "pq_m": [0, -1, 95], "pq_nbits": [0, -1, 17, 1.5],
            "id_bytes": [0, -1], "header_bytes": [-1], "extra_copies": [-1, 1.5],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    sb.estimate_dataset("fiqa", **{field: value})

    def test_invalid_plan_numbers(self):
        for field in ("free_bytes", "budget_bytes", "model_cache_bytes",
                      "build_temp_fraction", "build_temp_min_bytes", "free_reserve_bytes",
                      "ceph_replicas", "packed_record_bytes", "deployment_overhead_bytes"):
            for value in (-1, float("nan"), float("inf")):
                kwargs = {"free_bytes": FREE_BYTES, field: value}
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    sb.estimate_plan(["fiqa"], **kwargs)

    def test_codebook_uses_override_dimensions_and_bits(self):
        item = sb.estimate_dataset("fiqa", dim=384, pq_m=48, pq_nbits=4)
        self.assertEqual(item["assumptions"]["codebook_bytes"], 384 * 16 * 4)

    def test_deployment_requires_explicit_inputs_and_counts_replication(self):
        kwargs = dict(free_bytes=10**12, budget_bytes=10**12, scope="deployment",
                      corpus_overrides={"nq": 10**10}, free_reserve_bytes=10**9,
                      packed_record_bytes=128, ceph_replicas=3)
        self.assertFalse(sb.estimate_plan(["nq"], **kwargs)["download_ready"])
        plan = sb.estimate_plan(["nq"], deployment_overhead_bytes=12345, **kwargs)
        self.assertTrue(plan["download_ready"])
        self.assertEqual(plan["replicated_dataset_bytes"],
                         3 * (plan["datasets_total_bytes"] + plan["packed_records_bytes"]))
        self.assertEqual(plan["packed_records_bytes"], sb.DATASET_SIZES["nq"]["docs"] * 128)
        self.assertEqual(plan["required_free_bytes"], plan["estimated_total_bytes"] + 10**9)
        self.assertEqual(plan["datasets"]["nq"]["assumptions"]["corpus_source"], "explicit override")

    def test_reserve_and_missing_budget_refuse(self):
        plan = sb.estimate_plan(["fiqa"], free_bytes=FREE_BYTES,
                                budget_bytes=FREE_BYTES, free_reserve_bytes=FREE_BYTES)
        self.assertFalse(plan["fits_free"])
        self.assertTrue(plan["refusals"])
        self.assertFalse(sb.estimate_plan(["fiqa"], free_bytes=FREE_BYTES)["download_ready"])

    def test_cli_good_invalid_and_refusal(self):
        script = str(pathlib.Path(sb.__file__).resolve())
        cases = [
            (["--scope", "deployment", "--datasets", "nq", "--corpus-bytes", "nq=10000000000",
              "--free-reserve-bytes", "1000000000", "--deployment-overhead-bytes", "1000000000",
              "--packed-record-bytes", "128"], 0),
            (["--datasets", "hotpotqa"], 1),
            (["--budget-bytes", "-1"], 2),
            (["--model-cache-bytes", "-1"], 2),
            (["--build-temp-fraction", "nan"], 2),
            (["--corpus-bytes", "nq=bad"], 2),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, script, "--free-bytes", str(10**12),
                     "--budget-bytes", str(10**12)] + args,
                    capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, expected, result.stderr)
                if expected != 2:
                    self.assertEqual(json.loads(result.stdout)["download_ready"], expected == 0)

    def test_packed_codes_round_per_record_and_changed_docs_unknown(self):
        item = sb.estimate_dataset("fiqa", docs=3, dim=6, pq_m=3, pq_nbits=1)
        self.assertEqual(item["components"]["pq_codes_bytes"], 3)
        self.assertEqual(item["estimate_status"], "lower_bound")

    def test_override_does_not_bypass_budget_or_large_policy(self):
        plan = sb.estimate_plan(["nq", "hotpotqa"], free_bytes=10**12,
                                budget_bytes=1, corpus_overrides={"nq": 10**10, "hotpotqa": 10**10})
        self.assertIn("both large corpora", " ".join(plan["refusals"]))
        self.assertFalse(plan["fits_budget"])

    def test_pilot_dataset_estimate_within_20_percent(self):
        plan = sb.estimate_plan(
            ["scifact", "fiqa"], free_bytes=FREE_BYTES, model_cache_bytes=0
        )
        error = abs(plan["estimated_total_bytes"] - PILOT_DATASETS_ACTUAL) / float(
            PILOT_DATASETS_ACTUAL
        )
        self.assertLessEqual(error, 0.20, "dataset footprint error %.3f" % error)

    def test_pilot_total_with_one_model_within_20_percent(self):
        plan = sb.estimate_plan(["scifact", "fiqa"], free_bytes=FREE_BYTES)
        error = abs(plan["estimated_total_bytes"] - PILOT_TOTAL_ACTUAL) / float(
            PILOT_TOTAL_ACTUAL
        )
        self.assertLessEqual(error, 0.20, "pilot total error %.3f" % error)

    def test_components_follow_documented_formula(self):
        item = sb.estimate_dataset("fiqa", extra_copies=2)
        docs = 57_638
        self.assertEqual(item["components"]["vectors_bytes"], docs * sb.DIM * 4)
        self.assertEqual(item["components"]["pq_codes_bytes"], docs * sb.PQ_M)
        self.assertEqual(item["components"]["ids_bytes"], docs * 8)
        self.assertEqual(
            item["components"]["extra_copies_bytes"],
            2 * item["components"]["encoded_bytes"],
        )

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            sb.estimate_plan(["does-not-exist"], free_bytes=FREE_BYTES)


if __name__ == "__main__":
    unittest.main()
