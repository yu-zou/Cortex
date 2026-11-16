# Resident IVF-PQ Comparison

`compare.py` uses real Faiss `IndexIVFPQ.search_preassigned`, exported from the
same serialized index with the same PQ codebook, codes, queries, residual policy,
list IDs, vector IDs and final top-k. It supports D=768, M=96, 8-bit PQ, squared
L2 and plain L2 coarse quantization. Transforms, filtering, duplicate IDs and
oversized lists are rejected. Empty lists are skipped. Independent float64 ADC
validates list candidates and final merged top-k; ambiguous numerical cutoffs
fail explicitly. Exported bundles are rederived from the index when loaded.

From the checkout root, using a Python environment containing Faiss and NumPy:

```sh
python hw/faiss_compare/compare.py export --index INDEX.faiss --queries queries.npy --output hw/faiss_compare/fixture_export --k 5 --nprobe 4
python hw/faiss_compare/compare.py benchmark --bundle hw/faiss_compare/fixture_export/matched.npz --threads 1 --warmup 3 --repetitions 20
python -m unittest discover -s hw/host_benchmark -p test_host.py -v
python -m unittest discover -s hw/faiss_compare -p test_compare.py -v
```

CPU data is resident in host DDR after bundle load. Setup, independent
correctness validation and warmup are outside timing. Timed calls include output
allocation and exclude coarse selection. CPU thread count is explicit. FPGA API
`stage` preloads selected lists into FPGA DDR; `matched_fpga_search` scans the same
lists with residual queries, validates each local top-k, merges and validates the
final top-k. The transport shares one exclusive transaction lock and quarantine
state. The embedding owner must serialize staging and searches. Every search
reloads codebook from resident DDR into BRAM; no unverified reuse optimization.

FPGA per-list timing separates query transfer, control, observed launch-to-done,
result transfer and end-to-end. Batch validated time also includes reference
validation and merge. Polling time is not kernel time. Kernel time and speedup
remain null. Mock and unverified owner transport labels are preserved in reports.
There is no hardware benchmark CLI or CPU fallback labeled as FPGA.

See [owner and ABI contract](../host_benchmark/OWNER_PROTOCOL.md). Hardware remains
blocked pending safe owner implementation, kernel VFIO lifecycle evidence,
device-access authorization and bitstream/register-map verification. No device
access or FPGA measurements have been performed. Previous offline IndexPQ-only
comparison was not reused as an IVF-PQ baseline.
