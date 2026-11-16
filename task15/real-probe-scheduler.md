# Task15: Real FiQA Probe to Scheduler Trace Adapter

**Status**: Adapter delivered; frozen inputs missing locally.  
**Date**: 2026-11-03

## Summary

Strict local adapter from frozen actual FiQA probes/cluster sizes to C++ scheduler trace CSV input. No synthetic substitution; missing inputs cause explicit failure. Uniform arrival replay only; service costs document source (ideal geometry or declared proxy). Preserves query order, probe IDs, and cluster sizes exactly.

## Available Inputs

**MISSING LOCALLY**: The following frozen FiQA files are required but not found:

1. **Probes JSONL** (`*.probes.jsonl`)
   - One JSON object per line
   - Fields: `query_id`, `centroid_ids`, `cluster_sizes`, `candidate_count`
   - Must match probe_contract.py schema
   - Expected location: Unknown

2. **Index Metadata JSON** (`*metadata.json`)
   - IndexMetadata fields: `index_namespace`, `index_version`, `dataset`, `encoder_model`, `encoder_revision`, `dimension`, `metric`, `normalized`, `total_centroids`, `nprobe`
   - Expected location: Unknown

3. **Cluster Sizes JSON** (`*cluster_sizes.json`)
   - JSON array of integers, length = total_centroids
   - Each entry is cluster size for that centroid ID
   - Expected location: Unknown

4. **Encoder Config JSON** (`*encoder_config.json`)
   - Fields: `model`, `revision`, `dimension`, `normalize_embeddings`
   - Must match metadata exactly
   - Expected location: Unknown

**Search attempted**:
- `/mnt/d/RAG/GitHubPrj/Cortex/model/evidence/` - not found
- `/mnt/d/RAG/GitHubPrj/Cortex/.omo/evidence/` - exists but no FiQA probes
- `/home/user/OpenCodePrj/.omo/evidence/` - task15 artifacts but no FiQA data
- System-wide search for `*.probes.jsonl`, `*fiqa*.jsonl` - no results

## Adapter Implementation

**Location**: `/mnt/d/RAG/GitHubPrj/Cortex/model/scripts/fiqa_to_scheduler_trace.py`

### Contract Validation

Uses existing `probe_contract.py` validators:
- Exact query ID preservation
- No duplicate queries or cluster IDs
- Dimension/hash/count matching
- Cluster size table consistency
- All required subtasks present

### Arrival Model

**Uniform replay only** (first adapter):
- Rate specified as rational QPS
- Floor of absolute rational schedule: `arrival_ns = start_ns + (i * 1e9 * rate.denominator) // rate.numerator`
- Avoids accumulated rounding
- No replay from actual timestamps (separate future work)

### Service Cost Model

**Supported**: `ideal_pq_only`
- Source: Ideal uncalibrated geometry (from `real_probe_des.py`)
- Fixed + linear per candidate: `cycles = fixed_cycles + cycles_per_doc * candidate_count`
- Default: 180 MHz clock, 2160 fixed cycles (12 μs), 2 cycles/doc
- Scope: PQ-only ADC distance; **NO residual-IP, NO rerank, NO CPU ADC**

**NOT supported**: `ideal_residual_ip`
- Explicit rejection: DES does not compute coarse centroid contribution or residual correction
- Residual-IP requires actual coarse centroid vectors not in scheduler trace

### Output Format

**scheduler-trace-inputs.csv** (from `scheduler_trace_runner.cc:66-68`):
```
trace,seed,task_id,query_id,arrival_ns,service_ns,estimate_ns,estimate_label,task_class,query_class
```

- `trace`: Fixed `"fiqa_real"`
- `seed`: Fixed `0` (deterministic from frozen data)
- `task_id`: Sequential 1-indexed per subtask
- `query_id`: deterministic sequential positive uint64 assigned by first occurrence; full namespace/index-version/original-query identity is retained in the manifest `query_id_mapping` (structured tuple identity, no delimiter join)
- `arrival_ns`, `service_ns`, `estimate_ns`: Integer nanoseconds
- `estimate_label`: Fixed `"trace_supplied"` (perfect estimates supplied by the trace)
- `task_class`: `"cluster_{centroid_id}"`
- `query_class`: `"nprobe{N}_cands{C}"`

**Identity manifest JSON**:
- Input file SHA256 digests
- Metadata fields (namespace, version, dimension, encoder hash)
- Query/subtask counts
- Arrival model specification
- Service cost model with explicit source/scope/exclusions
- Validation flags (no duplicates, all subtasks, dimension match)

## Model Boundary

### Real Components
- Saved query-specific coarse probes (from actual embeddings)
- Cluster sizes from actual index build
- Query order preserved exactly
- Probe nprobe and candidate counts exact

### Modeled Components
- Uniform synthetic arrivals (not replay)
- Ideal PQ scan cost (uncalibrated)

### NOT Modeled (Out of Scope)
- Embedding computation
- Coarse search
- Query-level top-R selection/merge
- Raw vector reads
- Rerank (R64 quality cost)
- Network, queuing, routing
- Actual hardware execution

### Explicit Exclusions
- Residual-IP scoring (requires coarse centroids + residual PQ ADC)
- CPU ADC + modeled device cost mixing
- Test-based parameter selection
- Synthetic perfect services as measured hardware

## Usage

```bash
python3 /mnt/d/RAG/GitHubPrj/Cortex/model/scripts/fiqa_to_scheduler_trace.py \
  --probes <frozen_probes.jsonl> \
  --metadata <metadata.json> \
  --cluster-sizes <cluster_sizes.json> \
  --encoder-config <encoder_config.json> \
  --expected-queries <N> \
  --output-csv <trace.csv> \
  --output-manifest <manifest.json> \
  --rate-qps 10.0 \
  --start-ns 0 \
  --cost-model ideal_pq_only \
  --clock-ns 5.555556 \
  --fixed-cycles 2160.0 \
  --cycles-per-doc 2.0
```

**Exit codes**:
- 0: Success
- 1: Validation failure (missing file, wrong count, duplicate query, dimension mismatch, etc.)
- 2: Usage error

## Regression Tests

**Location**: `/mnt/d/RAG/GitHubPrj/Cortex/model/tests/test_fiqa_to_scheduler_trace.py`

**Coverage**:

### Negative Cases
1. Missing probes file
2. Missing metadata file
3. Missing cluster sizes file
4. Missing encoder config file
5. Zero expected_queries
6. Negative expected_queries
7. Dimension mismatch (metadata vs encoder)
8. Cluster size cardinality mismatch
9. Query count mismatch (actual vs expected)
10. Duplicate query IDs
11. Residual-IP cost model (explicitly rejected)
12. Unknown cost model
13. Non-uniform arrival kind
14. Negative rate_qps
15. Zero rate_qps
16. Negative start_ns

### Positive Cases
1. Correct subtask count (queries * nprobe)
2. Uniform arrival timing (floor of rational schedule)
3. Ideal PQ-only cost computation
4. CSV header format
5. CSV row format (matches scheduler_trace_runner.cc)
6. Complete end-to-end flow with small frozen fixture

**Test fixture**: 2 queries, nprobe=2, 3 centroids, matches all known counts

**Run tests**:
```bash
cd /mnt/d/RAG/GitHubPrj/Cortex/model
pytest tests/test_fiqa_to_scheduler_trace.py -v
```

## C++ Scheduler Integration

**Existing runner**: `/mnt/d/RAG/GitHubPrj/Cortex/sys/src/cortex/osd/scheduler_test/scheduler_trace_runner.cc`

Current runner uses hardcoded synthetic traces from `scheduler_trace_sim.h`:
- `uniform_equal`, `burst_mixed`, `long_starvation`, `multiquery_fanout`, `hand_counterexample`

**Integration options**:

### Option 1: External CSV Load Mode (Recommended)
Add minimal CSV loader to existing runner:
```cpp
std::vector<TraceTask> load_csv(const std::string& path);
```
- Preserves frozen trace generators
- No policy reimplementation in Python
- Existing 3CTest regression preserved

### Option 2: Python DES Wrapper
Use existing `exp_osd_sched.py` infrastructure:
- Already has SyntheticQuery, run_simulation, compute_metrics
- Can wrap C++ scheduler via Python bindings
- More invasive; may not preserve exact C++ test behavior

**Recommended**: Option 1 with minimal CSV load, bounded by existing capacity/aging constants, strict validation matching current trace requirements.

## Next Steps

1. **Locate frozen FiQA probe files**
   - Check remote artifact storage (SSH auth currently blocked)
   - Check BEIR dataset download cache
   - Regenerate if necessary from actual FiQA embeddings

2. **Run adapter once frozen files available**
   ```bash
   python3 fiqa_to_scheduler_trace.py \
     --probes <found_probes.jsonl> \
     --metadata <found_metadata.json> \
     --cluster-sizes <found_sizes.json> \
     --encoder-config <found_encoder.json> \
     --expected-queries <actual_count> \
     --output-csv /tmp/fiqa_trace.csv \
     --output-manifest /tmp/fiqa_manifest.json \
     --rate-qps 10.0
   ```

3. **Validate manifest**
   - Verify input SHA256 digests
   - Check query/subtask counts match expectations
   - Confirm dimension/hash/nprobe match known FiQA index

4. **Add CSV load to C++ runner** (if Option 1)
   - Parse CSV header, validate columns
   - Load tasks matching TraceTask struct
   - Reject overflow, missing columns, wrong types
   - Add 1 CTest with small fixture

5. **Run scheduler policies on real FiQA trace**
   ```bash
   ./scheduler_trace_runner /tmp/fiqa_output
   ```
   - Compare FIFO, SJF, aging, query-aware policies
   - Analyze wait time, latency, fairness on real probes
   - No claim of system P99/QPS or hardware execution

## Constraints Respected

- No SSH/remote attempts (auth blocked)
- No synthetic substitution for missing data
- No test-based parameter tuning
- Local commands bounded 60 seconds
- No packages/downloads
- Existing production code unchanged
- 3CTest regression preserved
- Service costs explicitly source-tagged
- Separate arrival trace from query content

## Evidence Appended

**Adapter source**: `/mnt/d/RAG/GitHubPrj/Cortex/model/scripts/fiqa_to_scheduler_trace.py` (371 lines)  
**Tests**: `/mnt/d/RAG/GitHubPrj/Cortex/model/tests/test_fiqa_to_scheduler_trace.py` (500+ lines)  
**This doc**: `/mnt/d/RAG/GitHubPrj/Cortex/task15/real-probe-scheduler.md`

**Missing inputs documented**:
- `*.probes.jsonl` - not found locally
- `*metadata.json` - not found locally
- `*cluster_sizes.json` - not found locally
- `*encoder_config.json` - not found locally

**Test commands** (once frozen files available):
```bash
# Run regression tests
cd /mnt/d/RAG/GitHubPrj/Cortex/model
python3 -m pytest tests/test_fiqa_to_scheduler_trace.py -v --tb=short

# Run adapter
python3 scripts/fiqa_to_scheduler_trace.py \
  --probes <path> --metadata <path> --cluster-sizes <path> --encoder-config <path> \
  --expected-queries <N> --output-csv /tmp/trace.csv --output-manifest /tmp/manifest.json \
  --rate-qps 10.0

# Verify manifest
jq '.query_count, .subtask_count, .metadata.dimension, .input_files' /tmp/manifest.json

# Check CSV format
head -3 /tmp/trace.csv
```

**No further matrix/strategy chains**: This deliverable completes the local adapter with explicit missing-input documentation as requested.

## Update 2026-11-03: Numeric query-ID mapping boundary

- CSV `query_id` is now a deterministic sequential positive integer by first occurrence, bounded to `UINT64_MAX`, matching the C++ selector's `uint64_t` task identity.
- Identity is the structured tuple `(index_namespace, index_version, original_query_id)`. Duplicate full identities are rejected before output; no delimiter join, delimiter parse, or randomized hash is used.
- Manifest `query_id_mapping` preserves `query_id`, `index_namespace`, `index_version`, `original_query_id`, `probe_index`, and `query_row`.
- CSV is generated with `csv.writer`; CSV label is `trace_supplied`; manifest service cost remains `declared_scan_proxy`.
- No C++ CSV loader was added or changed, and no C++ interoperability is claimed. Earlier partial C++ loader artifacts on disk remain unverified and outside this slice.

Verified from `/mnt/d/RAG/GitHubPrj/Cortex/model`:

```text
python3 -m unittest discover -s tests -p test_fiqa_scheduler_numeric.py -q
Ran 18 tests in 0.176s
OK

python3 tests/test_probe_contract.py -q
Ran 9 tests in 0.331s
OK
```

## Update 2026-11-03: Local adapter-to-selector interop

- Runner CLI is now strict: `scheduler_trace_runner <output-dir>` (2 argv) or `scheduler_trace_runner --input-csv <csv-path> <output-dir>` (4 argv). Other argument counts return nonzero.
- Both modes require an existing empty output directory; the CSV is fully validated before any output or loader manifest is written. Input CSV and existing results are never overwritten.
- The C++ loader parses numeric CSV `query_id` directly as uint64, preserving adapter query groups instead of hashing numeric strings.
- Python CSV writing rejects newline text before file creation and uses `csv.writer` for comma/quote escaping; the C++ parser rejects unterminated quotes.
- New standard-library integration test `model/tests/test_fiqa_scheduler_interop.py` runs the actual adapter CLI then the compiled runner on a synthetic 2-query/4-subtask fixture with costs 12112/13112 ns, checks all 4 arms, query groups, counts, conservation, and unchanged input hash. Negative cases isolate bad argv, numeric query ID, header, label, duplicate task ID, and nonempty output.
- Executed: CTest 4/4 PASS 0.07 s (loader 21 groups); numeric 18 PASS 0.189 s; probe contract 9 PASS 0.351 s; interop 6 PASS 0.205 s.
- Frozen real FiQA inputs and remote authentication remain unavailable. No real-FiQA replay, system P99/QPS, or hardware claim.

## Update 2026-11-03: CSV/JSON output escaping

- Runner now has reusable `csv_field` and `json_string` escaping applied to every variable text output in all four CSVs and `scheduler-traces.json`.
- CSV fields containing comma, quote, CR/LF are quoted with doubled quotes; JSON escapes quote, backslash, standard control escapes, and other control bytes as `\u00XX`.
- Numeric/time fields and policy/model/cost arithmetic are unchanged; newline input remains rejected before write/load.
- Interop regression fixture now contains comma, quote, backslash and a tab in trace/task/query class text; all generated CSVs are parsed with `csv.DictReader`/`csv.reader` (exact texts and column counts 10/14/12/24), JSON with `json.load`, plus grouping/counts/costs/conservation and unchanged input hash.
- Executed: CTest 4/4 PASS 0.07 s (loader 21 groups); numeric 18 PASS 0.180 s; probe contract 9 PASS 0.342 s; interop 7 PASS 0.197 s.
