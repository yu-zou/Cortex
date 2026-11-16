# Ceph loader / driver ABI correction

## Input availability and authority

`task8/build-audit.md` was requested first, but did not exist in this checkout
when work began (nor did `task8/`). There is no `driver/include/`; the actual
public driver header is `include/cortex_driver_api.h`. The existing offline
harness is `/tmp/opencode/cortex-task8-audit/CMakeLists.txt`.

Authority was established from that public header, the implemented export in
`driver/src/cortex_driver_exports.cc`, and the result population in
`driver/src/smartssd_driver.cc`. All agree on eight read arguments and document
address/length results. The Ceph copy disagreed while reporting the same version.

| Contract item | Original driver | Original Ceph | Corrected shared ABI |
|---|---|---|---|
| Version | 1 | 1 | 2 |
| Read arguments | 8 (last: search mode) | 7 | 8 |
| Entry fields | distance, doc_addr, doc_length, reserved | vector_id, distance | driver fields |
| Entry size, host x86-64 | 24 | 16 | 24 |
| Distance offset | 0 | 8 | 0 |
| Result size | 12008 | 8008 | 12008 |
| Result status/count/entries offsets | 0 / 4 / 8 | 0 / 4 / 8 | 0 / 4 / 8 |
| Capacity | 500 | 500 | 500 |
| Metrics | L2=0, IP=1 | L2=0, IP=1 | unchanged |

The 24-byte entry is a naturally aligned **host** structure, not a packed
128-bit FPGA word. No packing or driver-result reinterpretation was introduced.

## Changes

- Ceph's public header now forwards to the authoritative repository-level header.
  Loader function members derive their types from its declarations; the read
  `dlsym` cast derives from the member rather than repeating the signature.
- ABI version is 2 because two incompatible version-1 binaries already existed.
  Version checking still precedes init and symbol use. Version 1 is rejected.
- Loader header uses real Ceph `include/common_fwd.h` rather than pulling in the
  full performance-counter implementation; this permits compiling the actual
  loader without generated Ceph configuration. No substitute Ceph headers used.
- PrimaryLogPG supplies explicit `CORTEX_MODE_IVFPQ` and transfers document
  address, length, distance and status. It no longer reads nonexistent vector IDs.
- Semantic result and entry wire envelopes are both version/compatibility 2;
  decoders reject old vector-ID payloads, including an empty old result. RGW
  retains document length through merge and emits `doc_addr`, `doc_length`, and
  `distance`, rather than mislabeling addresses as `id`.
- The Python ctypes caller supplies all eight arguments, declares their widths,
  and checks version 2 before driver init. Existing C++ direct callers already
  supply eight arguments and rebuild against the shared header. The existing
  driver ABI test's explicit version expectation is updated to 2.
- No parser, KS, residual, search implementation, or metric ordering changes.

## Executed red evidence (before production edits)

After adding the regression sources/CMake targets, with the original production
headers still present:

```sh
cmake -S /tmp/opencode/cortex-task8-audit -B /tmp/opencode/cortex-task8-audit/build
cmake --build /tmp/opencode/cortex-task8-audit/build --target loader_contract_ceph -j2
```

Observed exit 2. Compiler diagnostics included:

```text
static assertion failed: ambiguous version 1 must be rejected
static assertion failed: semantic_read must take all eight arguments
static assertion failed                  [sizeof(cortex_topk_entry) == 24]
static assertion failed                  [offsetof(..., distance) == 0]
'struct cortex_topk_entry' has no member named 'doc_addr'
'struct cortex_topk_entry' has no member named 'doc_length'
'struct cortex_topk_entry' has no member named '_pad'
static assertion failed                  [sizeof(cortex_read_result) == 12008]
```

This is a compile-time red, not an intentionally executed undefined-behavior
call into the old undersized return buffer. The new eight-argument dynamic test
also cannot compile against the original seven-argument loader member.

During migration, the first green attempt exposed an additional stale literal:
`DriverABITest.ApiVersion` expected `1u` while the export returned 2. Updated that
expectation and repeated the relevant suites successfully.

## Executed green evidence

```sh
cmake --build /tmp/opencode/cortex-task8-audit/build --target test_loader_abi test_driver_abi -j2
ctest --test-dir /tmp/opencode/cortex-task8-audit/build -R '^(test_loader_abi|test_driver_abi)$' -V
bash -n /mnt/d/RAG/GitHubPrj/Cortex/test/integration/entrypoint.sh
/tmp/opencode/cortex-task8-audit/build/bin/test_loader_abi --gtest_output=xml:/tmp/opencode/cortex-task8-audit/abi-fix-green.xml
```

Observed: both CTest targets passed, 6 existing driver ABI cases plus 3 new loader
cases. Shell syntax check exited 0. XML records 3 loader tests, 0 failures.

The same contract source is compiled independently against the driver include
path and the Ceph include path, preventing a shared include guard from hiding a
disagreement within a single translation unit. Static assertions cover read and
other exported function types, size/alignment/offsets, signed status, unsigned
count, 500-entry capacity, and metric constants.

New runtime cases:

1. Actual loader rejects a version-1 DSO with `-ENOTSUP` before init (the DSO's
   init aborts if called), leaving the loader unloaded and read pointer null.
2. Actual loader loads a transport-only DSO compiled against the driver header.
   The DSO checks all eight argument values, including last argument
   `0x13579bdf` (not default IVFPQ=0), top-k=500, and both metrics. Checks all 500
   entries including index 499, high-bit 64-bit document addresses, lengths,
   distances, reserved tail fields, success and negative status, and canaries
   before/after the result. Checks cache exports and unload pointer clearing.
3. Independently `dlopen`s the built **real driver** and checks version 2 and
   its uninitialized eight-argument read error (`-EINVAL`, count 0).

The nondefault mode is a transport sentinel only; it does not claim the real
driver implements any mode other than IVFPQ.

## Exact coverage limitations / migration

- Actual loader `.cc` compiled and exercised; actual driver export built and
  exercised. Existing driver ABI lifecycle tests use the harness's existing
  mock hardware library. The new transport DSO is explicitly a call-contract
  stub, **not Ceph integration or hardware validation**.
- Full Ceph build is unavailable. PrimaryLogPG, semantic result serialization,
  and RGW changes were source-inspected but not compiled or run here. Wire
  compatibility rejection and HTTP result shape still require parent/full-Ceph
  verification. The integration shell was syntax-checked, not executed.
- Rebuild/redeploy driver, OSD and RGW together. Do not mix old/new binaries:
  new loaders reject every version-1 driver (even the otherwise authoritative
  old driver); old version-checking loaders reject version 2. External callers
  must check version 2 and supply eight arguments with the shared result layout.
- OSD/RGW semantic result wire version 2 is intentionally incompatible with
  version-1 vector-ID responses. Coordinate clients/servers; external HTTP
  consumers must migrate from `id` to `doc_addr` plus `doc_length`. No reliable
  address-to-vector-ID conversion exists in this code, so none is invented.
- The forwarding header requires the repository-level `include/` tree when
  packaging/building `sys/` independently.
- The review-work skill was loaded, but its required native multi-agent review
  tools/agent roles are unavailable in this toolset. Its five-lane review is
  **inconclusive**, not approved. Parent verification remains required.

No git commands, commits, downloads, or remote hardware were used. This evidence
does not mark the parent plan complete.
