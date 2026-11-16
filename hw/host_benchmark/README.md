# Resident IVF-PQ Host Comparison

All commands below are software only. No device paths are opened. VFIO is
unconditionally blocked because closing/releasing a VFIO device may reset it or
its bus even without an explicit reset ioctl. This package performs no binding,
PCI config writes, engine programming, SSD operations or reset recovery.

Run with an environment containing NumPy and CPU Faiss:

```sh
python -m unittest -v test_benchmark
python compare.py export --index trained.faiss --queries queries.npy --list-ids list_ids.npy --output exported --k 10
python compare.py benchmark --bundle exported/matched.npz --threads 1 --warmup 3 --repetitions 20
```

Run from this directory. On this workstation the available interpreter is
`python3` (Faiss 1.15.0).
An existing trained plain `IndexIVFPQ` is required: D=768, M=96, 8 bits, squared
L2, unique nonnegative IDs, at most 8192 entries per list. No transforms,
polysemous filtering or max_codes budget is accepted. Queries are float32 Nx768;
list IDs are integer NxNprobe with no duplicate or invalid lists per query.
Alternatively omit `--list-ids` and specify `--nprobe`: coarse assignment then
happens once during export and is excluded from both resident search timings.
Empty lists are exported and skipped on the FPGA side. Selected candidate count
must be at least k (1..500).

The bundle contains the serialized real IVF index, original queries, selected
list IDs, coarse distances, exact PQ codebook, codes, vector IDs and per-list
residual queries. It is never re-encoded or retrained. On load, all derived arrays
are revalidated against the serialized index. The manifest hashes all exports.
`search_preassigned` scans these identical lists in host DDR. A separate float64
ADC implementation checks top-k IDs and distances before timed searches. An
ambiguous top-k boundary is rejected rather than used for a speedup claim.
Distance tolerance is rtol=1e-5, atol=1e-3. This is a numerical consistency check
on PQ search, not a recall claim against original full precision vectors.

Each selected list is exported twice: `historical` has a 64-byte header and
128-byte records, with a separate codebook. `source-inline` inserts the codebook
after the header, as required by the current `hw/src/acc_top.cpp`. Code bytes
follow the actual reversed beat slices consumed by `adc_engine`; IDs occupy
record bytes 88..95 and doc_len at 84..87 is zero. Residual subtraction occurs
on the host once per selected list, using the same coarse centroid as Faiss.
This implements full preassigned IVF search through per-list scans plus host
top-k merging, rather than calling an archived synthetic cluster a full IVF.

## Host API And Blockers

`OwnedXdmaTransport` borrows already-open H2C/C2H character descriptors from an
exclusive owner; it never opens, duplicates or closes them. The owner must
independently verify descriptor direction, device identity, synchronous MM DMA
semantics and driver reset behavior, and supply explicit confirmations. This
cannot be established safely by this task, so there is no hardware CLI. No
actual driver or hardware behavior has been validated. DMA calls can block in
the driver; the polling timeout does not cancel a blocked DMA syscall.

`Session` implements only the historical register map from
`host_xdma_test/vfio_xdma_acc.c`: separate codebook pointer, reuse register,
start at 0x70 and 64-byte result stride. `abi=host.ABI` is an explicit assertion
that the owner verified the loaded bitstream. Current PR HLS source instead
uses an inline codebook, no codebook/reuse arguments, and 16-byte result stride.
Its generated AXI-Lite register map and loaded bitstream identity are not
available here. Its control path therefore remains blocked; do not assert the
historical ABI for that source. A source-compatible data export is not proof
that its register ABI or implementation works on the board.

For a separately verified historical image, `compare.stage(transport, data,
abi=host.ABI)` stages all selected lists at disjoint DDR addresses and returns
sessions. `compare.matched_fpga_search(sessions, data)` passes the correct
per-list query, scans min(k,list_size), validates local results, merges, and
validates global top-k. Sessions share the query/result buffers and must be
used serially. Codebook reload reads resident FPGA DDR each call. Staging must
complete before resident timing. Transfer/control failures poison the session;
no reset, retries or further operations follow a timeout. Re-stage/recovery
requires the device owner's intervention. The stream result sink must also be
known to drain; a backpressured AXIS sink can prevent completion.

## Timing Scope

CPU setup (including file load and bundle validation), correctness, and resident
batch search are separate. CPU transfer is zero because data is already in host
DDR. Reported search includes the Python call and result allocation. Neither
CPU timing nor FPGA resident scope includes coarse list selection. Faiss thread
count, warmups, repetitions, version and bundle hash are recorded.

FPGA sessions report setup transfer, query transfer, control, result transfer,
launch-to-done observation and end-to-end time separately. Polling includes DMA
register-read latency and sleep overhead; it is **not kernel timing**. Kernel
time stays null until a verified hardware cycle counter or timestamp source is
available. Matched batch time currently includes correctness checks, host
residual-query serialization, per-list launches and merge; it is explicitly
labeled and must not be compared to CPU search time as a speedup. All speedup
fields remain null. Real matched hardware results and a verified timing source
are concrete blockers to reporting any FPGA speedup. Mock results are protocol
tests only and must never be reported as hardware performance measurements.
