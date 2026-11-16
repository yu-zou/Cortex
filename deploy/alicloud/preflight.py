"""Standard-library-only, offline approval/schema gate; never runs Terraform or APIs."""
import argparse
import datetime as dt
import ipaddress
import json
import math
import re
import sys


FIELDS = set("approved approval_ref region zone vpc_cidr subnet_cidr ssh_cidr "
             "ssh_range_approved key_name experiment expires_at ttl_hours budget_total "
             "budget_hourly currency quote_ref residual_allowance cleanup_plan daemon_image "
             "cortex_abi ceph_commit pool_size pool_min_size failure_domain nodes disks".split())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, minimum=0, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    require(value > minimum if positive else value >= minimum, name + " out of range")
    return value


def inspect_config(c, now=None):
    require(isinstance(c, dict) and set(c) == FIELDS, "missing/unknown deployment fields")
    require(c["approved"] is True, "explicit approval required")
    for key in ("approval_ref", "region", "zone", "key_name", "experiment", "currency",
                "quote_ref", "cleanup_plan"):
        require(isinstance(c[key], str) and c[key].strip(), key + " required")
    require(re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", c["experiment"]), "invalid experiment tag")
    private = [ipaddress.ip_network(x) for x in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
    vpc, subnet, ssh = [ipaddress.ip_network(c[k], strict=True) for k in
                        ("vpc_cidr", "subnet_cidr", "ssh_cidr")]
    require(vpc.version == subnet.version == ssh.version == 4, "IPv4 only")
    require(any(vpc.subnet_of(p) for p in private) and subnet.subnet_of(vpc), "private VPC/subnet required")
    require(ssh.prefixlen >= 28 and (ssh.prefixlen == 32 or c["ssh_range_approved"] is True),
            "SSH requires /32 or explicitly approved /28 or narrower")
    require(type(c["ssh_range_approved"]) is bool, "SSH approval must be boolean")
    ttl = number(c["ttl_hours"], "ttl_hours", positive=True)
    hourly_limit = number(c["budget_hourly"], "budget_hourly", positive=True)
    total_limit = number(c["budget_total"], "budget_total", positive=True)
    residual = number(c["residual_allowance"], "residual_allowance")
    expiry = dt.datetime.fromisoformat(c["expires_at"].replace("Z", "+00:00"))
    now = now or dt.datetime.now(dt.timezone.utc)
    require(expiry.tzinfo is not None and (expiry - now).total_seconds() >= ttl * 3600,
            "expiry must cover TTL from preflight time")
    require(c["cortex_abi"] == 2 and type(c["cortex_abi"]) is int, "Cortex ABI2 required")
    require(re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", c["daemon_image"]), "immutable daemon digest required")
    require(re.fullmatch(r"[0-9a-f]{40}", c["ceph_commit"]), "Ceph fork commit required")
    require(c["failure_domain"] == "host", "only host CRUSH failure domain supported")
    require(isinstance(c["nodes"], dict) and c["nodes"], "nodes required")
    roles, osd_hosts, hourly = set(), set(), 0
    for name, node in c["nodes"].items():
        require(re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", name), "invalid node name")
        require(isinstance(node, dict) and set(node) == {"roles", "instance_type", "image_id", "system_disk_gib", "hourly"}, "invalid node schema")
        require(isinstance(node["roles"], list) and node["roles"] and
                set(node["roles"]) <= {"mon", "mgr", "gateway", "client", "osd"}, "invalid roles")
        roles.update(node["roles"])
        if "osd" in node["roles"]:
            osd_hosts.add(name)
        require(all(isinstance(node[k], str) and node[k].strip() for k in ("instance_type", "image_id")),
                "explicit approved type/image required")
        require(type(node["system_disk_gib"]) is int and node["system_disk_gib"] >= 40, "system disk >=40 GiB")
        hourly += number(node["hourly"], "node hourly", positive=True)
    require(roles == {"mon", "mgr", "gateway", "client", "osd"}, "all Ceph/client roles required")
    require(any("client" in n["roles"] and "osd" not in n["roles"] and "gateway" not in n["roles"]
                for n in c["nodes"].values()), "dedicated load client required")
    require(isinstance(c["disks"], dict), "disk map required")
    disk_hosts = set()
    for name, disk in c["disks"].items():
        require(re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", name), "invalid disk name")
        require(isinstance(disk, dict) and set(disk) == {"node", "purpose", "size_gib", "category", "performance_level", "hourly", "initialize"}, "invalid disk schema")
        require(disk["node"] in osd_hosts and disk["purpose"] == "bluestore" and
                disk["initialize"] is False, "dedicated uninitialized BlueStore disk required")
        require(type(disk["size_gib"]) is int and disk["size_gib"] >= 40, "data disk >=40 GiB")
        require(disk["category"] == "cloud_essd" and disk["performance_level"] in {"PL0", "PL1", "PL2", "PL3"}, "explicit ESSD performance level required")
        disk_hosts.add(disk["node"])
        hourly += number(disk["hourly"], "disk hourly", positive=True)
    require(disk_hosts == osd_hosts, "every OSD needs its own distinct data disk")
    for key in ("pool_size", "pool_min_size"):
        require(type(c[key]) is int and c[key] > 0, key + " must be positive integer")
    require(c["pool_min_size"] <= c["pool_size"] <= len(osd_hosts), "replicas exceed actual OSD host domains")
    total = hourly * ttl + residual
    require(hourly <= hourly_limit and total <= total_limit, "expected cost exceeds approved budget")
    return {"status": "OFFLINE_VALID", "cloud_plan": False, "expected_hourly": hourly,
            "expected_total": total, "currency": c["currency"], "ttl_hours": ttl,
            "nodes": c["nodes"], "dedicated_disks": c["disks"],
            "expiry": c["expires_at"], "cleanup_plan": c["cleanup_plan"],
            "runtime_config": {k: c[k] for k in ("daemon_image", "cortex_abi", "ceph_commit",
                                                 "pool_size", "pool_min_size", "failure_domain")},
            "runtime_status": "NOT_INSTALLED_NOT_FORMATTED; task8 binaries required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tfvars", help="JSON containing only the deployment object")
    args = parser.parse_args()
    try:
        with open(args.tfvars) as stream:
            data = json.load(stream)
        require(set(data) == {"deployment"}, "expected only deployment tfvar")
        print(json.dumps(inspect_config(data["deployment"]), indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print("PREFLIGHT_REFUSED: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
