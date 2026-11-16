"""Synthetic approvals only: these tests never authorize a deployment."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

from preflight import inspect_config


def fixture():
    return {
        "approved": True, "approval_ref": "SYNTHETIC-TEST-ONLY",
        "region": "cn-hangzhou", "zone": "cn-hangzhou-h",
        "vpc_cidr": "10.42.0.0/16", "subnet_cidr": "10.42.1.0/24",
        "ssh_cidr": "192.0.2.1/32", "ssh_range_approved": False,
        "key_name": "existing-test-key", "experiment": "synthetic",
        "expires_at": "2099-01-01T00:00:00Z", "ttl_hours": 2,
        "budget_total": 10, "budget_hourly": 3, "currency": "CNY",
        "quote_ref": "synthetic-not-a-price", "residual_allowance": 2,
        "cleanup_plan": "archive; review state-bound destroy; reconcile disks snapshots EIPs billing",
        "daemon_image": "registry.invalid/cortex@sha256:" + "a" * 64,
        "cortex_abi": 2, "ceph_commit": "c712a30e040f19a01c511da16ba6ffd22cbe9083",
        "pool_size": 1, "pool_min_size": 1, "failure_domain": "host",
        "nodes": {
            "storage": {"roles": ["mon", "mgr", "gateway", "osd"],
                        "instance_type": "ecs.approved-test", "image_id": "approved-image-test",
                        "system_disk_gib": 40, "hourly": 1},
            "client": {"roles": ["client"], "instance_type": "ecs.approved-test",
                       "image_id": "approved-image-test", "system_disk_gib": 40, "hourly": 1}},
        "disks": {"osd0": {"node": "storage", "purpose": "bluestore",
                            "size_gib": 40, "category": "cloud_essd",
                            "performance_level": "PL1", "hourly": 0.1,
                            "initialize": False}},
    }


class PreflightTests(unittest.TestCase):
    def test_offline_inventory(self):
        result = inspect_config(fixture())
        self.assertEqual(result["status"], "OFFLINE_VALID")
        self.assertFalse(result["cloud_plan"])
        self.assertAlmostEqual(result["expected_hourly"], 2.1)
        self.assertAlmostEqual(result["expected_total"], 6.2)

    def reject(self, change):
        config = copy.deepcopy(fixture())
        change(config)
        with self.assertRaises(ValueError):
            inspect_config(config)

    def test_no_approval(self):
        self.reject(lambda c: c.update(approved=False))

    def test_zero_budget(self):
        self.reject(lambda c: c.update(budget_total=0))

    def test_broad_ssh(self):
        self.reject(lambda c: c.update(ssh_cidr="0.0.0.0/0", ssh_range_approved=True))

    def test_narrow_range_requires_approval(self):
        self.reject(lambda c: c.update(ssh_cidr="192.0.2.0/28"))
        config = fixture()
        config.update(ssh_cidr="192.0.2.0/28", ssh_range_approved=True)
        self.assertEqual(inspect_config(config)["status"], "OFFLINE_VALID")

    def test_bad_diskplan(self):
        for field, value in [("initialize", True), ("node", "client"),
                             ("purpose", "root"), ("size_gib", 0)]:
            self.reject(lambda c: c["disks"]["osd0"].update({field: value}))

    def test_missing_osd_disk(self):
        self.reject(lambda c: c.update(disks={}))

    def test_budget_hourly_total_ttl(self):
        for changes in [dict(budget_hourly=1), dict(budget_total=1),
                        dict(ttl_hours=0), dict(residual_allowance=-1),
                        dict(expires_at="2000-01-01T00:00:00Z")]:
            self.reject(lambda c: c.update(changes))

    def test_no_public_or_unknown_fields(self):
        self.reject(lambda c: c.update(public_ui=True))
        self.reject(lambda c: c.update(access_key="not-a-key"))

    def test_abi_digest_and_replication(self):
        for changes in [dict(cortex_abi=1), dict(daemon_image="upstream:latest"),
                        dict(pool_size=3), dict(pool_min_size=0)]:
            self.reject(lambda c: c.update(changes))

    def test_private_cidr(self):
        self.reject(lambda c: c.update(vpc_cidr="8.0.0.0/8"))
        self.reject(lambda c: c.update(subnet_cidr="10.43.0.0/24"))

    def test_cleanup_and_quote_required(self):
        for key in ["cleanup_plan", "quote_ref", "approval_ref"]:
            self.reject(lambda c: c.update({key: ""}))

    def test_nonfinite_and_bool_prices(self):
        for value in [float("nan"), float("inf"), True, -1]:
            self.reject(lambda c: c["nodes"]["client"].update(hourly=value))

    def test_cli_exit_codes_and_no_cloud_plan(self):
        script = pathlib.Path(__file__).with_name("preflight.py")
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "synthetic.json"
            config = fixture()
            path.write_text(json.dumps({"deployment": config}))
            result = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["cloud_plan"])
            for key, value in [("approved", False), ("budget_total", 0),
                               ("ssh_cidr", "0.0.0.0/0")]:
                bad = copy.deepcopy(config)
                bad[key] = value
                path.write_text(json.dumps({"deployment": bad}))
                result = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("PREFLIGHT_REFUSED", result.stderr)


if __name__ == "__main__":
    unittest.main()
