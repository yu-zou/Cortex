import pathlib
import os
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.exp_osd_sched import STRATEGIES
from scripts.unified_sched_matrix import SCHEMA, run_matrix
from strata_sim.contracts.schema import ClockKind, ProfileRef
from strata_sim.des.fpga import CalibrationPoint, FpgaServiceModel


def fixture_fpga_model() -> FpgaServiceModel:
    """Build a calibrated test model without claiming formal hardware evidence."""
    model = FpgaServiceModel(
        profile=types.SimpleNamespace(profile_id="fixture-profile"),
        ref=ProfileRef(
            profile_id="fixture-profile",
            clock_kind=ClockKind.DES_LOGICAL,
            source="test fixture",
            evidence_id="unified-matrix-contract-test",
        ),
        clock_ns=1.0,
        topk_init_cycles=10.0,
        process_cycles_per_doc=2.0,
        push_cycles_per_result=0.0,
    )
    model.calibrate(
        [CalibrationPoint(n_docs=1, top_k=1, total_cycles=12.0, origin="fixture")]
    )
    return model


class UnifiedSchedMatrixContractTests(unittest.TestCase):
    def test_all_strategies_share_trace_and_emit_complete_arms(self):
        payload = run_matrix(
            project_root=pathlib.Path(__file__).resolve().parents[1],
            workloads=("uniform",),
            seeds=(7,),
            num_osds_values=(1,),
            cache_mib_values=(1,),
            num_queries=8,
            num_tenants=2,
            num_clusters=4,
            nprobe=2,
            vectors_per_cluster=8,
            fpga_model=fixture_fpga_model(),
        )

        self.assertEqual(payload["schema"], SCHEMA)
        self.assertEqual(len(payload["arms"]), len(STRATEGIES))
        self.assertEqual(
            {arm["strategy"] for arm in payload["arms"]}, set(STRATEGIES)
        )
        digests = {arm["trace_digest_sha256"] for arm in payload["arms"]}
        self.assertEqual(len(digests), 1)
        self.assertEqual(payload["trace_records"][0]["digest_sha256"], digests.pop())
        for arm in payload["arms"]:
            self.assertEqual(arm["metrics"]["samples"], 8)
            self.assertEqual(arm["metrics"]["strategy"], arm["strategy"])
            self.assertTrue(arm["metrics"]["claims"]["same_trace_across_strategies"])
            self.assertEqual(
                arm["metrics"]["claims"]["model_evidence"], "injected_fixture"
            )

    def test_four_osd_matrix_uses_three_replicas(self):
        payload = run_matrix(
            project_root=pathlib.Path(__file__).resolve().parents[1],
            workloads=("uniform",),
            seeds=(7,),
            num_osds_values=(4,),
            cache_mib_values=(1,),
            strategies=("fifo",),
            num_queries=2,
            num_tenants=1,
            num_clusters=2,
            nprobe=1,
            vectors_per_cluster=8,
            fpga_model=fixture_fpga_model(),
        )
        placement = payload["arms"][0]["placement"]
        self.assertTrue(all(len(osds) == 3 for osds in placement.values()))

    def test_result_affecting_environment_is_rejected(self):
        os.environ["ADAPT_CB"] = "1"
        try:
            with self.assertRaisesRegex(ValueError, "ADAPT_CB"):
                run_matrix(
                    project_root=pathlib.Path(__file__).resolve().parents[1],
                    workloads=("uniform",),
                    seeds=(7,),
                    num_osds_values=(1,),
                    cache_mib_values=(1,),
                    strategies=("fifo",),
                    num_queries=1,
                    fpga_model=fixture_fpga_model(),
                )
        finally:
            del os.environ["ADAPT_CB"]

    def test_duplicate_matrix_values_are_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            run_matrix(
                project_root=pathlib.Path("/definitely/missing/project"),
                workloads=("uniform",),
                seeds=(7, 7),
                num_osds_values=(1,),
                cache_mib_values=(1,),
                strategies=("fifo",),
                num_queries=1,
            )

    def test_formal_builder_still_fails_closed_without_artifacts(self):
        with self.assertRaisesRegex(ValueError, "HLS profile not found"):
            run_matrix(
                project_root=pathlib.Path("/definitely/missing/project"),
                workloads=("uniform",),
                seeds=(7,),
                num_osds_values=(1,),
                cache_mib_values=(1,),
                num_queries=1,
            )


if __name__ == "__main__":
    unittest.main()
