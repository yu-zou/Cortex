"""CRUSH-style placement for the 20-OSD topology (5 hosts x 4 OSDs).

Properties modelled (same as Ceph CRUSH for a replicated pool):
  * deterministic: cluster -> PG -> OSDs by hashing, no state;
  * failure-domain aware: the 3 replicas land on 3 distinct hosts;
  * stable: adding/removing an OSD only moves the PGs that map to it;
  * uniform: per-OSD replica load within ~+/-13% at 1024 PGs.

Implementation: rendezvous (highest-random-weight) hashing, which is the
equal-weight limit of CRUSH's straw2 bucketing.  It is *not* a bit-exact port of
Ceph's rjenkins/straw2, but it reproduces the placement statistics that matter
for the simulation.  Not modelled: PG splitting/merging, weighted OSDs,
rebalancing and recovery traffic.
"""
from __future__ import annotations

M64 = (1 << 64) - 1


def _mix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def crush_placement(num_osds: int, cluster_ids, replication: int = 3,
                    osds_per_host: int = 4, n_pg: int = 4096):
    """Return {cluster_id: ReplicaSet} with host-diverse, hash-based placement."""
    from strata_sim.sched.rgw_scheduler import ReplicaSet

    n_hosts = (num_osds + osds_per_host - 1) // osds_per_host
    # Pre-compute the replica set of every PG once.
    pg_sets: dict[int, tuple[int, ...]] = {}
    for pg in range(n_pg):
        order = sorted(range(num_osds), key=lambda o: _mix(pg * 64 + o), reverse=True)
        chosen: list[int] = []
        hosts: set[int] = set()
        for o in order:
            h = min(o // osds_per_host, n_hosts - 1)
            if h in hosts:
                continue
            hosts.add(h)
            chosen.append(o)
            if len(chosen) == replication:
                break
        pg_sets[pg] = tuple(chosen)

    out = {}
    for cid in cluster_ids:
        pg = _mix(int(cid)) % n_pg
        osds = pg_sets[pg]
        out[int(cid)] = ReplicaSet(cluster_id=int(cid),
                                   osd_ids=tuple(sorted(osds)),
                                   primary_osd_id=osds[0])
    return out
