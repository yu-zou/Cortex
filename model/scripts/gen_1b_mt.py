#!/usr/bin/env python3
"""1B multi-tenant spec with per-tenant nlist = 2*sqrt(S).

  * 30 small private tenants:  S ~ log-uniform(1e6, 5e6)
  * 13 medium private tenants: S ~ log-uniform(1e7, 5e7)
  * 250 shared tenants:        one public index of 5e8 docs
  * total ~= 1.0e9 vectors

Only sizes and cluster counts are generated (the DES never reads vectors).
Cluster sizes follow the empirical MS MARCO-9M IVF shape, rescaled per tenant.
Recall is computed separately from the 9M seed vectors.
"""
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "workload_1b_mt"
OUT.mkdir(parents=True, exist_ok=True)

N_SMALL, N_MED, N_SHARED = 30, 13, 250
PUB_DOCS = 500_000_000
NPROBE = 32
rng = np.random.RandomState(20261023)

base = np.load(ROOT / "data/derived/workload_v2/spec_window.npz", allow_pickle=True)
names = [str(x) for x in base["tenant_names"]]
assert len(names) == N_SMALL + N_MED + N_SHARED

shape = np.bincount(
    np.load(ROOT / "data/derived/msmarco_9m/assign.npy").astype(np.int64), minlength=4096
).astype(np.float64)
shape = np.sort(shape) / shape.mean()


def gen(total, nlist):
    q = (np.arange(nlist) + 0.5) / nlist
    v = np.quantile(shape, q)
    v = np.maximum(np.round(v / v.sum() * float(total)).astype(np.int64), 1)
    v[np.argmax(v)] += int(total) - int(v.sum())
    return v


# tenants with the most trace events become the private (paying) ones
import collections
import json

cnt = collections.Counter()
for i, line in enumerate(open(ROOT / "data/derived/trace/events.jsonl")):
    if i >= 500:
        break
    cnt[json.loads(line)["tenant"]] += 1
ranked = [t for t, _ in cnt.most_common()]
priv_names = ranked[: N_SMALL + N_MED]
small_set = set(priv_names[:N_SMALL])

cls = np.zeros(len(names), dtype=np.int64)
sizes, offs, nlists, S = [], [0], [], []
for nm in names:
    if nm not in priv_names:
        cls[names.index(nm)] = 1           # shared tenant -> public index
        S.append(0)
        nlists.append(0)
        sizes.append(np.zeros(0, dtype=np.int64))
        offs.append(offs[-1])
        continue
    if nm in small_set:
        s = int(round(10 ** rng.uniform(6, np.log10(5e6))))
    else:
        s = int(round(10 ** rng.uniform(7, np.log10(5e7))))
    nl = int(max(16, round(2.0 * np.sqrt(s))))
    v = gen(s, nl)
    S.append(s)
    nlists.append(nl)
    sizes.append(v)
    offs.append(offs[-1] + len(v))

priv = np.concatenate(sizes)
pub = gen(PUB_DOCS, int(round(2.0 * np.sqrt(PUB_DOCS))))
np.savez(
    OUT / "spec_1b_mt.npz",
    tenant_names=np.array(names),
    S=np.array(S, dtype=np.int64),
    nlist=np.array(nlists, dtype=np.int64),
    offsets=np.array(offs),
    private_sizes=priv,
    pub_sizes=pub,
    cls=cls,
    params=np.array(
        "1b_mt: %d small + %d medium private, %d shared on a %d-doc public index (nlist=2sqrt(S))"
        % (N_SMALL, N_MED, N_SHARED, PUB_DOCS)
    ),
)
tot = int(priv.sum()) + int(pub.sum())
print(
    "private tenants=%d  private docs=%.3fB  public docs=%.3fB  TOTAL=%.3fB"
    % (len(priv_names), priv.sum() / 1e9, pub.sum() / 1e9, tot / 1e9)
)
print(
    "private clusters=%d (median %d docs/cluster, max %d)"
    % (len(priv), int(np.median(priv)), int(priv.max()))
)
print(
    "public clusters=%d (median %d docs/cluster)"
    % (len(pub), int(np.median(pub)))
)
print("nlist: min=%d median=%d max=%d" % (min(n for n in nlists if n), int(np.median([n for n in nlists if n])), max(nlists)))
