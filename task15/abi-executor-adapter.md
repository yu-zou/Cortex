# Task15: C ABI CPU executor adapter

2026-11-01. **Locally exercised integration slice; remote acceptance blocked.**
No PG async enablement or Ceph async claim.

## Starting state and scope

Read the approved lifecycle design and both cloud-hardware notepads, production
executor/preflight and tests, authoritative `include/cortex_driver_api.h`, Ceph
forwarding header, result guard, and loader implementation. The interrupted work
already left `semantic_cpu_backend.{h,cc}`, three adapter tests, typed executor
results, and their production CMake source registration. These were preserved
and extended. No source baseline was reconstructed with git.

The current slice uses value-owned query/object vectors and explicit cache ID.
Result entries retain document address, length and FP32 distance; checked admission
charges `sizeof(SemanticResultEntry)` and verifies actual vector capacities.
Construction now rejects capacity above `CORTEX_TOPK_MAX` before allocating result
storage. The existing private completion destructor and custom deleter remain:
input/result/completion allocation is freed before charge/callback retirement.

The adapter validates ID 1..4096, finite query, dimension <=768, nonempty object,
metric, explicit IVFPQ search mode and preallocated output capacity. It invokes
the actual eight-argument ABI, validates status/count before indexing, rejects
nonfinite/padded entries, and publishes count only after success and cancellation
checks. No packed 16-byte FPGA casts or PG cache-getter calls are used. SHA256
immutable byte binding remains worker-local and is not persistent cache generation.

## Lifetime contract

The constructor requires both a function and a non-null shared API owner. That
owner must exclusively own driver initialization and the DSO, releasing shutdown /
dlclose only in its final destructor. Backend retains the owner; backend and
transport must outlive executor join, as required by the existing executor API.
No raw-function-only constructor remains. A shared alias to the existing singleton
loader does **not** meet this contract: its public unload remains independent.
Loader leasing and OSD shutdown integration are still pending.

The real-driver test supplies a DSO owner with final shutdown/dlclose. The barrier
test drops the initiating owner and cancels a running call: task/byte charges and
API owner remain live, the call reads its original owned input after release,
join completes, and cancellation publishes zero count. No detach, forced call
interruption or bounded-shutdown promise is made. CPU finite-return attestation
applies only to the intended bounded CPU implementation, not arbitrary ABI DSOs.

## Actual test-first evidence

Before production fixes, executed regressions failed:

* `RejectResultCapacityAboveAbiBeforeAllocation`: capacity 501 incorrectly
  constructed successfully instead of throwing `length_error`.
* `UnsupportedSearchModeRejectedBeforeCall`: mode 1 incorrectly dispatched once
  and published count 2 instead of rejecting before dispatch.

These outputs were observed in this session. Metadata/lease/real-driver tests
were added later; no claim of historical red coverage for every test.

The requested `/tmp/opencode/cortex-task8-audit` and SSH socket were absent (the
temporary directory was empty). Recreated the isolated CMake harness using the
existing production hw/driver test registrations and local installed GTest,
OpenSSL and libstdc++; no downloads/packages. Local compiler is **G++9.4**, not
the requested remote G++12. Initial local dependency/link/search-path failures
were resolved only in the isolated harness/environment.

Actual final execution:

```
timeout 180 cmake --build /tmp/opencode/cortex-task8-audit/build -j2
timeout 180 ctest --test-dir /tmp/opencode/cortex-task8-audit/build --output-on-failure --output-junit /tmp/opencode/cortex-task8-audit/green.xml
```

* Regression suite excluding adapter: **16/16 CTests PASS**, then new adapter
  target **1/1 PASS**; final combined **17/17 PASS**, 0.74 seconds.
* Executor **20/20 GTests PASS**: all original 18 retained, plus ABI capacity and
  exact typed-entry budget boundary tests. Allocation/deallocation probes enabled.
* Adapter **8/8 GTests PASS**, including seven controlled ABI result modes:
  excessive fixed count, excessive caller count, infinity, reserved padding,
  negative error, invalid positive error and NaN. Also output capacity, invalid
  identity/search mode, immutable binding, cancellation and lease validation.
* Actual exported CPU driver + actual CPU mock DSO, synthetic KS256 compact-v2
  residual/IP fixture through executor: three equal scores 0.75, 64-bit document
  address and distinct lengths preserved. Output tie order is compared with direct
  actual ABI output, not a newly imposed document sort.
* Checked-in standalone `backend_test/CMakeLists.txt` also configured/built and
  passed **2/2 CTests**, using `CMAKE_PREFIX_PATH=/home/user/miniconda3`, local
  `-L/home/user/miniconda3/lib` executable link flag and
  `LD_LIBRARY_PATH=/home/user/miniconda3/lib` at test execution.
* Isolated harness including both builds: **19,620 KiB allocated**, below 256 MiB.
  Local root free **789,954,748,416 bytes**, above 10 GiB. Every build used -j2
  and a 180-second build timeout. No remote build space measurement claimed.

## Files changed in this session

Relative to Cortex:

1. `sys/src/cortex/osd/semantic_cpu_backend.h`
2. `sys/src/cortex/osd/semantic_cpu_backend.cc`
3. `sys/src/cortex/osd/semantic_executor.h`
4. `sys/src/cortex/osd/semantic_executor.cc`
5. `sys/src/cortex/osd/test_semantic_cpu_backend.cc`
6. `sys/src/cortex/osd/test_semantic_executor.cc`
7. `sys/src/cortex/osd/backend_test/CMakeLists.txt` (new)
8. `task15/abi-executor-adapter.md` (new)

Temporary harness: `/tmp/opencode/cortex-task8-audit/CMakeLists.txt`.
Existing production `osd/CMakeLists.txt` source registration was already present
and was not edited. OSD/PG/RGW/loader/hw/driver runtime sources were not edited;
ABI2, KS256 and residual behavior remain covered by the exercised regressions.

## Remaining acceptance/integration

Actual SSH command with BatchMode and the specified ControlPath to
`chenming@192.168.103.229` returned **Permission denied (publickey,password)**.
No remote sync/build/test was performed. Must restore SSH and execute the remote
G++12/configured-Ceph compile and existing immutable first-ten offline FiQA ABI
fixture. Neither that run nor a new preflight run is claimed here.

Future work: loader-owned leasing/unload exclusion, OSD service initialization,
PG preflight/admission and OpFinisher continuation, registry/generation/reset,
normal/fast shutdown ordering and real Ceph lifecycle tests. Existing default
synchronous runtime remains in place. No timing-wait feature was added.

Review-work skill loaded; required Oracle/unspecified-high/native spawn tools
are unavailable. Five review lanes are **INCONCLUSIVE**, not approval. Local
tests pass but this slice does not satisfy the remote acceptance requirement yet.
No plan checkboxes, cluster/resources, authorization claims or commits changed.

## Remote verification continuation — 2026-11-01

**The historical SSH blocker above is resolved. Remote GCC12 focused acceptance
and first-ten offline adapter/executor verification PASS.** This does not approve
PG integration, loader leasing or whole-Ceph async lifecycle.

Host `chenming-ThinkStation-P7`, `/usr/bin/g++-12` Ubuntu 12.3.0. Created the
previously absent exclusive directory
`/home/chenming/cortex-source-build-v1/backend-validation-v1` (`$R`). Staged exact
current adapter/executor/tests and shared ABI/result-guard headers, current CPU
mock/driver sources and the checked-in backend CMake. `source-manifest.json`
records SHA256 for every staged production source; all hashes were checked after
execution. No baseline source, binary or runtime directory was overwritten.

Remote GTest was not installed as a package. The isolated harness compiled the
existing real vendored Google Test implementation at
`/home/chenming/cortex-build/sys/src/fmt/test/gtest/gmock-gtest-all.cc`, using its
headers and a conventional InitGoogleTest/RUN_ALL_TESTS main. No fake assertion
framework, package installation, downloads or FetchContent. All 28 tests ran;
allocation/deallocation instrumentation and subprocess fatal-publication cases
were enabled, with zero skips. Current mock and driver were compiled into new
shared libraries in `$R/build`; positive paths use these actual implementations.

### Actual matrix

| Remote check | Result |
|---|---|
| Focused CTest | 2/2 PASS, 0.16 s |
| Executor GTest XML | 20/20 PASS; original 18 plus two capacity/budget cases |
| Adapter GTest XML | 8/8 PASS; real compact-v2 path plus controlled error cases |
| Runtime ABI layout | ABI2; host entry 24 B; typed owned entry 16 B; ABI result 12,008 B |
| First ten FiQA through adapter **and executor** | 10/10 PASS; 320 calls, 3,200 list hits |
| Ordered global top10 vs saved Faiss/reconstruction references | 10/10 exact each |
| Max abs score error vs saved Faiss | 4.1764920044595755e-7 |
| Immutable objects and input hashes before/after | All 64 objects and declared inputs match |
| Second fresh-driver execution | PASS; hit output byte-identical |

FiQA first-ten matrix (each row: 32 calls, 320 hits, exact ordered top10):

| Query ID | Max abs error vs saved Faiss |
|---|---:|
| 10034 | 2.984802551231347e-7 |
| 10039 | 2.387998657704671e-7 |
| 10109 | 4.1764920044595755e-7 |
| 10122 | 2.9832086179126804e-7 |
| 10136 | 1.4877983095518132e-7 |
| 10137 | 3.281348266437334e-7 |
| 10152 | 1.1947099304299158e-7 |
| 10183 | 1.792004394873814e-7 |
| 10213 | 3.581227416837862e-7 |
| 10246 | 2.3839042662121557e-7 |

Existing fixture reused read-only:
`/tmp/opencode/fiqa-abi-validation-20261101/fixture`, manifest SHA256
`837d346c245f796f9a1476255cf505a08fa33868e5b5d09a3259ee93befcb016`.
Saved `results-final/queries.json` SHA256
`24e7470e595f557482f85c85f25ce293e334f34be8e57680f51bcdb385eacc3b`.
The fixture's historical hash-derived IDs exceed the adapter's range. This
isolated fresh driver explicitly maps immutable list L to cache ID L+1, recorded
in `fiqa-cache-mapping.json`; original IDs/fixture bytes are unchanged. Audited
KS256, residual-v2 flags, D768/M96, all 57,638 global row tokens exactly once,
zero lengths, original query input hash and full raw-query IP convention. The
runner feeds the saved 32 probes per query and compares against the saved Faiss
and independent-reconstruction goldens with `1e-5 + 1e-5*abs(reference)` tolerance;
it does not claim a newly recomputed Faiss/reconstruction oracle. Returned
metadata is checked and global merging uses `(negative_IP,row_token)`.

The verification executable prints layout sizes, creates a real driver owner,
and submits each list read through `SemanticCompletion` and `SemanticExecutor`.
It joins each task before consuming its completion. This is offline adapter
correctness, not concurrency throughput, timing calibration or a PG lifecycle.

### Reproduction and retained evidence

Remote `$R` holds `CMakeLists.txt`, `source-manifest.json`, staged `source/`,
`gtest_main.cc`, `fiqa_adapter.cc`, `verify-backend-fiqa.py`, `configure.log`,
`build.log`, `ctest.xml`, `executor.xml`, `adapter.xml`, `fiqa-summary.json`,
`fiqa-cache-mapping.json`, `fiqa-requests.txt`, `fiqa-hits.txt`, `fiqa-stderr.txt`
and `rerun-v1/` with the second run's independent outputs.

```sh
R=/home/chenming/cortex-source-build-v1/backend-validation-v1
cmake -S "$R" -B "$R/build" -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 -DCMAKE_BUILD_TYPE=Release
timeout 180 cmake --build "$R/build" -j2
timeout 180 ctest --test-dir "$R/build" --output-on-failure
timeout 180 "$R/build/test_semantic_executor" --gtest_output=xml:"$R/executor-parent.xml"
timeout 180 "$R/build/test_semantic_cpu_backend" --gtest_output=xml:"$R/adapter-parent.xml"
# Output directory must not exist; never overwrite prior run artifacts.
timeout 180 /home/chenming/strata_sim/.venv/bin/python "$R/verify-backend-fiqa.py" "$R/parent-rerun-v1"
```

Both runs' `fiqa-hits.txt` SHA256:
`3b6278aaeb3a4e77d3238adf12e1f7a3462edbbfa54288bca8697f39a5299c6e`.
New binary hashes:

* driver: `a76d116202f845b20cd90f4a6f7521bb1653373ebdca90291dc7898ec9138c45`
* mock: `b7839ffcc7dfc3759db98d4d0e907a44be5070d456b8275e9a6fffa89175669a`
* FiQA adapter runner: `69451f854eefea45faed5b11f622b80aa9e40a2f859fa17cc2509bddff247686`

Final directory allocated **12,480 KiB**, including both runs, under 256 MiB.
Free root after the first run **68,233,400,320 B**, above 10 GiB reserve. Builds
used `-j2` and 180-second timeout; tests/runners also bounded at 180 seconds.
Final process inspection found no active `$R/build/` worker. Small JSON/XML
evidence copied locally to `/tmp/opencode/{fiqa-summary,source-manifest}.json`
and `/tmp/opencode/{ctest,executor,adapter}.xml`.

Only this evidence document and notepads changed in the source workspace during
the continuation; verification harnesses were written under `/tmp/opencode` and
transferred into the exclusive directory. No PG/OSD/loader source modifications.
This was a standalone GCC12 build on the configured-Ceph host, not a rebuild of
the full configured Ceph target. Loader-owned lease/unload exclusion and all PG /
OSD integration and lifecycle acceptance remain pending as described above.
