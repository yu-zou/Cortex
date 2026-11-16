"""Dispatch must not do hypothetical work for cache-independent policies."""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path.cwd()))
sys.path.insert(0, str(pathlib.Path.cwd() / 'scripts'))
from exp_osd_sched import SyntheticQuery, run_simulation, build_fpga_model
from strata_sim.sched.osd_timing import OSDTimingModel
from strata_sim.des.fpga import FpgaServiceModel


class DispatchPerformanceTest(unittest.TestCase):
    def test_fifo_no_hypothetical_estimates_or_disabled_prefetch_peek(self):
        from exp_osd_sched import OsdScheduler
        queries = [SyntheticQuery(str(i), 't', 0, 100000000, (0,), 10, 'cb')
                   for i in range(8)]
        with patch.dict('os.environ', {'OSD_PREFETCH': '0'}), \
             patch.object(OSDTimingModel, 'estimate_service_ns',
                          side_effect=AssertionError('unused FIFO estimate')), \
             patch.object(OsdScheduler, 'peek',
                          side_effect=AssertionError('disabled prefetch peek')):
            result = run_simulation(queries, strategy='fifo', num_osds=3,
                                    window_width=8,
                                    fpga_model=build_fpga_model(pathlib.Path.cwd()))
        self.assertEqual(len(result[0]), 8)


if __name__ == '__main__':
    unittest.main()
