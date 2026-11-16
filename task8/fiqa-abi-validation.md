# Task 8: actual FiQA offline ABI2 validation

Recorded 2026-11-01. **PASS for this bounded CPU correctness slice: all 648
official FiQA test queries, nprobe=32, top10.** This is actual shared-library
C ABI execution against the current residual-IP CPU mock. Task 8 integration
remains incomplete; this establishes no Ceph-cluster, FPGA, cloud, latency,
throughput, or full-precision retrieval-quality claim.

## Implementation and staging

New files:

- `model/scripts/fiqa_cpucompact.py`: explicit existing-Faiss-to-CPQ1v2 packer.
- `model/scripts/validate_fiqa_abi.py`: eight-argument ctypes ABI2 runner,
  actual Faiss PQ reference, independent vector reconstruction oracle.
- `model/tests/test_fiqa_abi.py`: seven format, ID, cache, and actual ABI tests.

Workstation source data: `/home/chenming/strata_sim/data/benchmarks/fiqa`.
New isolated workstation directory:
`/tmp/opencode/fiqa-abi-validation-20261101` (abbreviated `$R` below).
Only the current mock/driver sources and headers were copied there, then
built with existing `/usr/bin/g++-12` 12.3.0, C++17,
`-O2 -shared -fPIC -pthread` (driver additionally `-ldl`). Mock sources:
`hw/src/{mock_smartssd,cortex_hw_mock_exports}.cc`; driver sources:
`driver/src/{smartssd_driver,driver_manager,cortex_driver_exports}.cc`.
Include paths are `$R/hw/include`, `$R/include`, and the respective source
directory. Existing Python 3.10.12 / NumPy 2.2.6 / Faiss 1.15.0:
`/home/chenming/strata_sim/.venv/bin/python`.

Local Python had no Faiss. The existing SSH socket was absent; established
authentication succeeded with host-key checking and an in-memory password
pipe, creating `/tmp/opencode/fiqa-abi-ssh`. Both libraries were compiled on
the workstation from the copied current sources, avoiding cross-host ABI
dependency assumptions. No package/model installation or index training.

Artifacts in `$R`:

- `fixture/manifest.json`: every input and every one of 64 object byte hashes,
  list sizes and deterministic cluster IDs; `fixture/list-00.cpq` ... `list-63.cpq`.
- `results-final/summary.json`, `results-final/queries.json`: authoritative
  final run; per-query probes, candidates, ordered IDs and all three top10 scores.
- `provenance.json`: copied source/script/header/binary SHA256 values,
  compiler/platform details and full document-normalization verification.
- `results/`: preserved initial successful run before adding post-run input
  hash rechecks. Final run reproduced identical per-query output hash.

Local evidence copies (small JSON only):
`/tmp/opencode/fiqa-abi-validation-20261101/{fixture-manifest,summary,queries,provenance}.json`.
Original source data and existing build outputs were not overwritten. Output
directories must be new; output files use exclusive creation.

## Verified index, format, and score contract

Packer checks actual `IndexIVFPQ`, trained, D768, 57,638 rows, 64 lists,
M96, nbits8, KS256, dsub8, code_size96, `by_residual=True`, and
`metric_type=faiss.METRIC_INNER_PRODUCT`. Faiss IP enum **0** is explicitly
translated to Cortex IP enum **1**. All lists' original codes, global IDs,
PQ centroids and reconstructed coarse centroids are packed without reencoding.
Global IDs must cover exactly `0..57637`, once each, across all lists.

CPQ1v2 little-endian layout:

| Range | Contents |
|---|---|
| 0..63 | M, D, N, CPQ1 magic, version2, KS256, flags1; zero reserved |
| 64..786495 | 786,432-byte FP32 PQ codebook, M/KS/dsub order |
| 786496..789567 | 3,072-byte inline FP32 coarse centroid |
| 789568 onward | **112-byte** record: 96 code bytes + uint64 row token + uint64 zero length |

All 64 object bytes total **56,987,808**. These are CPUcompact objects,
not the 128-byte hardware format. `doc_addr` deliberately carries the global
embedding **row token**, mapped through `document_ids.jsonl` to BEIR document
ID; `doc_length=0`. It is not silently interpreted as a text byte address.
Host result entry is independently checked as 24 bytes, result as 12,008 bytes.

Queries are already unit-L2-normalized: all648 norm range
`0.9999998807907104 .. 1.0000001192092896`. Full document embeddings were
also independently scanned: norm range
`0.9999998211860657 .. 1.0000001192092896`, shape `(57638,768)`.
No normalization is applied by these scripts. PQ reconstructions are not
renormalized. Driver receives the full query, not a query residual.

For each selected list the ABI returns top10, then Python merges by
ascending `(negative_IP, global_row_token)`. The score is
`-q dot (coarse_center + concat(PQ[m, code[m]]))`.
Faiss returns positive IP, largest first. The independent oracle explicitly
reconstructs full float32 vectors and computes float64 dot products over
**all candidates in the same 32 lists**, rather than copying LUT arithmetic.
Every returned list-local score is also checked against reconstruction.
Faiss `search_preassigned` uses the identical coarse probes; its output must
also agree with ordinary `index.search` at nprobe32.

Fixed prior-plan tolerance:
`abs_err <= 1e-5 + 1e-5*abs(reference)`. Ordered IDs must match exactly;
boundary score gaps are recorded per query and ties do not silently waive
an ID mismatch. Qrels are used only to verify the official test-ID set/order.

First ten official rows, lexicographically sorted test query IDs:
`10034, 10039, 10109, 10122, 10136, 10137, 10152, 10183, 10213, 10246`.
All ten are included in the full648 run.

## Results

| Check | Actual result |
|---|---:|
| Official queries verified | 648 / 648 |
| nprobe / per-list k / global k | 32 / 10 / 10 |
| Actual eight-argument `cortex_semantic_read` calls | 20,736 |
| Cold list loads / warm reads | 64 / 20,672 |
| Same-candidate reconstruction scores evaluated | 18,807,477 |
| Exact ordered top10 agreement with Faiss PQ | 648 / 648 |
| Exact ordered top10 agreement with reconstruction | 648 / 648 |
| Maximum global top10 absolute error versus Faiss | 8.940696716308594e-7 |
| Maximum all-list-hit absolute error versus reconstruction | 8.632421657495115e-7 |
| New tests, remote and local actual-library executions | 7 / 7 each |
| Existing isolated local CTests, parent rerun | 15 / 15 |

Final runner elapsed 56.3935 seconds is execution bookkeeping, not a
performance result. No correctness change to mock/driver was necessary.

Cache inspection (`driver/src/smartssd_driver.cc:163–183`) and an executable
test confirm a hit is keyed only by cluster ID: replacement object bytes,
even wrong-version bytes or changed centroids, are ignored. A fresh ID loads
the changed center, giving the expected changed score. There is no object
version/hash cache key and no reliable reload count exported by the shared API.
Cold/warm counts above are observed cache membership before calls, not a
claim about a hardware codebook-reload counter.

This run starts a fresh driver, holds all64 immutable objects, derives IDs
deterministically from full fixture identity plus list number, checks all64
32-bit IDs are distinct, and hashes objects before and after cached reads.
The fixture SHA256 identity is authoritative; the 32-bit namespace is not
a promise of universal cross-fixture collision freedom. Cross-fixture reuse
requires shutdown/new driver or explicit collision checks. Inputs are also
hash-checked before and after the final run.

Negative tests reject truncated header/payload, wrong format version, and
out-of-range row tokens during packing/parsing. The real driver independently
rejects truncated bytes and wrong versions on fresh IDs. The mock cannot
know the external corpus row bound; that bound is enforced by the fixture
and validator, not falsely attributed to the C ABI.

## Hashes

SHA256 (full byte-hash manifest additionally records sizes and all64 objects):

| Input | SHA256 |
|---|---|
| `ivfpq8.faiss` (6,978,084 B) | `5ff9df1f1b2ce00f0662ada5448c7babd85a2122d510da388a0784e8ac54341c` |
| `test_query_embeddings.npy` (1,990,784 B) | `a49c1ddcb9c7abd3c3183979ba4d5d75e3ca1d019b23f311457f6a3f202dd0ac` |
| `document_ids.jsonl` | `43a59c23fc70a636f2d9d95de01b7f75775ff24b0d06eb80be976e7031baa14a` |
| `test_query_ids.jsonl` | `60aa998bd28ed4713e7ca1911b7c1c72c681718ea73c8f1999b48614b217b73b` |
| `qrels/test.tsv` | `6adc2a640dcdd22bb8b3858f89107adef2a7c3db20a63550dfa7a0f71e379e44` |
| Dataset `manifest.json` | `c3f7c057e31cd0de12fc752f5467f6f436031dacffe0b948e946b6213a8c2c4d` |
| Document embeddings (normalization-only) | `32b4ed1dbb753739fcc38fe235f03e010b32062c6624e92d351900aeca00e03d` |

| Output / executable | SHA256 |
|---|---|
| Fixture manifest | `837d346c245f796f9a1476255cf505a08fa33868e5b5d09a3259ee93befcb016` |
| Per-query results | `24e7470e595f557482f85c85f25ce293e334f34be8e57680f51bcdb385eacc3b` |
| `libcortex_driver.so` | `82897bacf02d5f10ab9d9b35d90e6b8aff03eee03b443892dddd6d492d74e042` |
| `libcortex_hw_mock.so` | `2697e8cf65f585c6b9984cb33a6d82cda2a5347477fe4e90c4cabbd6c8df453b` |
| Current copied `mock_smartssd.cc` | `232679971cd4f9812bc4393ca1fe4f908fe3b661ea017e5b70c8a92dc41b92e7` |
| Current copied `smartssd_driver.cc` | `f9ee9cff862ac75967b57e86f57db7ae617399824a096402e19076eea4d2c8a2` |

## Reproduction

With the existing workstation Python and freshly built matching libraries:

```sh
export R=/tmp/opencode/fiqa-abi-validation-20261101
export DATA=/home/chenming/strata_sim/data/benchmarks/fiqa
export PY=/home/chenming/strata_sim/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Choose new fixture/result directories on subsequent runs.
$PY "$R/fiqa_cpucompact.py" --data "$DATA" --output "$R/fixture-new"
FIQA_ABI_DRIVER="$R/libcortex_driver.so" FIQA_ABI_MOCK="$R/libcortex_hw_mock.so" \
  $PY "$R/test_fiqa_abi.py" -v
$PY "$R/validate_fiqa_abi.py" --data "$DATA" --fixture "$R/fixture-new" \
  --library "$R/libcortex_driver.so" --mock "$R/libcortex_hw_mock.so" \
  --queries 648 --output "$R/results-new"
```

Parent also rebuilt `/tmp/opencode/cortex-task8-audit/build` and ran
`ctest --test-dir /tmp/opencode/cortex-task8-audit/build --output-on-failure`:
15/15 passed. Seven new tests passed against those local ABI2 libraries too.

Post-implementation review skill was loaded. Its mandated five reviewer
types/native spawning tools are unavailable in this session, so independent
five-lane review is **INCONCLUSIVE, not approved**. Executed correctness
evidence above is distinct from that unavailable review. No task-completion,
pilot-change, commit, deployment, or full-Ceph-build claim is made.
