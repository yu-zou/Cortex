# Local Validation, 2026-11-12

Environment: `python3`, Faiss 1.15.0.
No device access, reset, PCI binding/configuration or hardware measurements.

Focused test commands in README passed: 5 host tests and 2 comparison tests.
Host tests inject failure at every staging/control/read operation and exercise
timeout, cancellation, connection quarantine, exclusive transaction ownership,
real Unix socket request/response and deadline expiration. Comparison tests use
a real Faiss residual IVF-PQ index with several selected lists, an empty list,
nontrivial IDs, bit-slice packing, merged final top-k and bundle corruption.
Scripted/software results are labeled mock and never generate a speedup.

Real resident CPU run used the existing workstation fixture:

```
./fixture/fixture_ivfpq.faiss
./fixture/queries.npy
```

D=768, M=96, 8-bit, 16 lists, 4096 vectors, residual squared L2,
32 queries, nprobe=4, k=5, one CPU thread, 3 warmups, 20 repetitions.
Export at `fixture_export_k5/matched.npz` (ignored generated artifact), SHA256
`99e200d8cc2438872d72e6899eefb1175bed56094d16146a239959903b9203c2`.
All final IDs and distances passed independent float64 ADC validation.

| Quantity | Observed milliseconds |
| --- | ---: |
| Bundle load/revalidation setup | 18.254 |
| Independent correctness validation | 349.156 |
| Resident batch minimum | 3.074 |
| Resident batch median | 3.277 |
| Resident batch p95 | 4.236 |

These are local CPU observations, not controlled publication measurements.
Timing includes Python/Faiss output allocation and excludes coarse selection,
setup and independent validation. Kernel/FPGA time and speedup are null.
k=10 export was correctly rejected for a numerically ambiguous top-k boundary;
no tolerance was loosened. An earlier incomplete `fixture_export` directory is
an ignored failed export and must not be used as a validated bundle.

Default host unittest discovery also encountered another agent's concurrent
`test_benchmark.py`, whose old `import compare` path fails after the comparison
moved to `hw/faiss_compare`. That file was not modified. The focused commands
explicitly select this task's test files; a full shared suite is not claimed.

Hardware remains blocked by no device authorization, unverified bitstream ABI
and missing reviewed persistent hardware owner. Owner RPC is a real transport
client with bounded I/O, not a hardware DMA implementation. Direct VFIO is
deliberately disabled because first-open/last-close reset scope is not proven
safe. See the owner protocol for the exact lifecycle and OSD bridge contract.
