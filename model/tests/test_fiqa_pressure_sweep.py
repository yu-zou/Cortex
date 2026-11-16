import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.fiqa_pressure_sweep import _unique_positive, run_sweep


class FiqaPressureSweepContractTests(unittest.TestCase):
    def test_unique_positive_rejects_duplicates_and_non_positive_values(self):
        self.assertEqual(_unique_positive((1, 2, 4), "budgets"), (1, 2, 4))
        for values in ((), (0,), (-1,), (1, 1)):
            with self.assertRaises(ValueError):
                _unique_positive(values, "budgets")

    def test_invalid_strategy_is_rejected_before_input_loading(self):
        with self.assertRaisesRegex(ValueError, "invalid strategy"):
            run_sweep(
                probes=None,
                metadata=None,
                cluster_sizes=None,
                encoder_config=None,
                expected_queries=1,
                rate_qps="1000",
                slo_ns=1,
                top_k=1,
                num_osds=1,
                window_width=1,
                strategies=("not-a-strategy",),
                pressure_budgets_mib=(1,),
                control_budget_gib=1,
            )

    def test_duplicate_strategy_is_rejected_before_input_loading(self):
        with self.assertRaisesRegex(ValueError, "non-empty and unique"):
            run_sweep(
                probes=None,
                metadata=None,
                cluster_sizes=None,
                encoder_config=None,
                expected_queries=1,
                rate_qps="1000",
                slo_ns=1,
                top_k=1,
                num_osds=1,
                window_width=1,
                strategies=("fifo", "fifo"),
                pressure_budgets_mib=(1,),
                control_budget_gib=1,
            )


if __name__ == "__main__":
    unittest.main()
