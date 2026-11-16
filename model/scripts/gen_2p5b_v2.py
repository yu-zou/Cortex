#!/usr/bin/env python3
"""2.5x scale of the 1B multi-tenant spec, with nlist = 2*sqrt(S) recomputed for
BOTH the private tenants and the shared public index.

  43 private tenants: S *= 2.5, nlist = 2*sqrt(S)
  250 shared tenants: public index 5e8 -> 1.25e9, nlist = 2*sqrt(1.25e9) = 70,710
"""
import pathlib

import numpy as np

ROOT = pathlib.Path("/home/chenming/strata_sim")
SRC = ROOT / "data/derived/workload_1b_mt/spec_1b_mt.npz"
OUT = ROOT / "data/derived/workload_2p5b"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 2.5

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


s = np.load(SRC, allow_pickle=True)
names = [str(x) for x in s["tenant_names"]]
cls = s["cls"].astype(np.int64)
priv0 = s["private_sizes"].astype(np.int64)
offs0 = s["offsets"].astype(np.int64)

PUB = 1_250_000_000
pub_nlist = int(round(2.0 * np.sqrt(PUB)))
pub = gen(PUB, pub_nlist)

sizes, offs, nlists, S = [], [0], [], []
for i, nm in enumerate(names):
    if cls[i]:
        S.append(0); nlists.append(0); sizes.append(np.zeros(0, dtype=np.int64)); offs.append(offs[-1]); continue
    lo, hi = int(offs0[i]), int(offs0[i + 1])
    total = int(round(priv0[lo:hi].sum() * SCALE))
    nl = int(max(16, round(2.0 * np.sqrt(total))))
    v = gen(total, nl)
    S.append(total); nlists.append(nl); sizes.append(v); offs.append(offs[-1] + len(v))

priv = np.concatenate(sizes)
np.savez(OUT / "spec_2p5b.npz",
         tenant_names=np.array(names), S=np.array(S, dtype=np.int64),
         nlist=np.array(nlists, dtype=np.int64), offsets=np.array(offs),
         private_sizes=priv, pub_sizes=pub, cls=cls,
         params=np.array("2.5x scale, nlist=2sqrt(S) for private and public"))
tot = int(priv.sum()) + int(pub.sum())
print("total=%.3fB  PQ set=%.1f GB  cache=%.1f GB  ratio=%.2f"
      % (tot / 1e9, tot * 128 / 1e9, 20 * 6 * 1024**3 / 1e9, tot * 128 / (20 * 6 * 1024**3)))
print("private clusters=%d p50=%d | public clusters=%d p50=%d"
      % (len(priv), int(np.median(priv)), len(pub), int(np.median(pub))))
