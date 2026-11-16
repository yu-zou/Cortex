# Explicit FPGA OSD bridge

This is a separate ABI2 DSO, `libcortex_fpga_driver.so`. It does not change the
existing CPU/mock driver, perform device I/O, load a bitstream, reset a board, or
fall back to CPU. It requires an explicitly configured, trusted local broker
that actually owns the accelerator. The broker is **not implemented by this
bridge**. Hardware execution and numerical correctness remain unverified until
that broker and the appropriate hardware profile pass integration validation.

Build and isolated socket-fixture tests (no Ceph or hardware required):

```sh
cmake -S driver/fpga -B ./build
cmake --build ./build
ctest --test-dir ./build --output-on-failure
```

Optional adapter compile validation against the inspected, read-only source:

```sh
cmake -S driver/fpga -B ./build -DCORTEX_CEPH_SOURCE_ROOT=./cortex-source/sys
cmake --build ./build
patch --dry-run -p1 -d ./cortex-source/sys -i ./driver/ceph/runtime_factory.patch
```

Configure the ABI2 loader with the absolute path to this DSO and this exact
closed JSON grammar, substituting the socket and a timeout from 1 to 60000 ms:

```json
{"backend":"fpga","socket":"/run/user/1000/cortex-fpga.sock","profile":"cpq1v2-residual-ip","timeout_ms":1000}
```

The other supported profile is `cpq1v2-residual-l2`. No defaults, extra fields,
escaped paths, library search, mock configuration, or automatic fallback exist.
The DSO reports ABI2 even though the legacy PR driver header still describes
ABI1. The 24-byte host result entry is distinct from packed 16-byte wire results.
Ordinary Ceph I/O never enters this DSO. Semantic writes return `-EOPNOTSUPP`.
No persistent cluster cache is used: identical numeric cluster IDs across pools,
PGs or object versions cannot reuse another object's device data.

## Ceph integration overlay

The clean PR pins `sys` to submodule c712a30e, but that submodule is not populated
on this workstation. The inspected source-build directory contains newer dirty
ABI2/executor/PrimaryLogPG/RGW integration, without git metadata. It was read only.
Do not copy that whole directory into this PR or present this overlay as an
already integrated Ceph build.

For a clean checkout containing the inspected ABI2 executor integration, install
`driver/ceph/semantic_fpga_backend.h` as `sys/src/cortex/osd/semantic_fpga_backend.h`,
and the two `include/cortex_fpga_*.h` headers as `sys/src/cortex/` headers.
Apply `driver/ceph/runtime_factory.patch` in that Ceph checkout. It changes only
the explicitly configured backend branch; the CPU factory, artifact attestation,
executor, asynchronous PG completion/cancellation and RGW list selection remain
in place. The adapter holds the loader's exclusive generation lease until the
executor joins, resolves its cancellation extension from the same mapped DSO,
and never unloads code from a PG callback. Deployment must pin/review the driver
and broker; a symbol name or ABI version is not a hardware attestation.

The inspected legacy synchronous PrimaryLogPG path calls ABI2 `semantic_read`
directly and gets a finite timeout but has no cancellation callback. Use the
asynchronous executor profile with this adapter for cooperative PG cancellation.
The existing reviewed CPU hash pins cannot attest a different DSO and must not
be modified to bless it. OSD startup policy must explicitly authorize the chosen
isolated hardware fixture before enabling admission. End-to-end Ceph startup
and policy integration are blocked by the absent clean ABI2 Ceph source.

## Transport Agent Contract

`include/cortex_fpga_protocol.h` is the canonical **CPF1 version 1** definition.
All fields are explicitly little endian; padding is zero. Unix `SOCK_STREAM`,
same effective UID enforced using `SO_PEERCRED`, filesystem socket in an
owner-controlled directory. This assumes a trusted same-UID broker; HELLO is a
profile negotiation, not cryptographic proof of execution. No network listener.
One client operation at a time; competing calls return `-EBUSY`. Each connection
has HELLO/reply then SEARCH/reply and is closed. Startup does HELLO only.

The 40-byte frame header and all payload offsets are defined in the header.
Replies must echo request ID, profile and opcode with the reply bit. IDs are
monotonic within the DSO lifetime. Error replies have negative errno, no payload.
HELLO must report hardware execution kind 1 and object layout 2. A mock/test
broker may be used only in explicitly labeled software fixtures. A real broker
must reject unsupported hardware/profiles with `-ENOTSUP`, never relabel mock
results or raw-query A6 L2 ADC as residual/IP execution.

SEARCH includes the exact immutable CPQ1v2 object (64-byte header; M=96, D=768,
ksub=256, version=2, by_residual=1; 786432-byte codebook, 3072-byte centroid,
112-byte records containing 96 PQ bytes, row token LE64, document length LE64).
This bridge limits lists to 8192 records, finite floats, unique tokens and lengths
1..UINT32_MAX. It sends the raw query. L2 requires residual query `q-centroid`;
IP requires full residual reconstruction scores including the coarse-centroid
term. IP results are descending similarities; L2 is ascending squared distance.
The broker must preserve exactly those semantics or reject the profile. It owns
conversion to the hardware reader layout, codebook layout and result decoding.
Historical A6 reader/result layouts are not the CPQ1v2 or CPF1 layout.

RGW selects lists and merges their Top-k. A broker executes exactly the supplied
list, returns `min(k,n)` distinct input row tokens and their exact lengths. The
bridge checks framing, profile, request ID, count, order, finite scores, input
membership, duplicate IDs and lengths before publishing any count. It cannot
validate numeric scores without a matching software oracle; hardware acceptance
requires same-index/reference comparisons across queries and selected lists.

## Ownership And Failure

The bridge owns only nonblocking CLOEXEC sockets and per-call host buffers. It
never owns DMA descriptors, MMIO, FPGA DDR or a board lock. The broker must hold
exclusive device ownership and enforce bounded, cancellable transfers itself.
The client's monotonic deadline covers connect, handshake, upload and reply;
poll slices are at most 10 ms. The remaining timeout is sent to the broker.
Shutdown atomically closes admission and active calls observe `-ESHUTDOWN`;
callers must join before unloading the DSO, as enforced by the executor lease.
Callbacks must not block or throw. No background threads or detached calls.

**Connection EOF is cancellation.** A broker must monitor EOF while working,
stop publishing results, and drain or quarantine in-flight hardware before
accepting another owner/request. It must never reuse buffers still accessed by
DMA, reset shared hardware, or assume a client timeout stopped the accelerator.
The bridge does not retry. Timeout, cancellation, disconnect, shutdown and bad
frames return an error with zero results; remote `-EIO`/`-ENOTSUP` propagate.
If a driver offers only unbounded synchronous DMA, the broker cannot claim this
contract; deployment is blocked until bounded device operations are available.

No physical hardware test, SSD write, device reset, Vivado job or replacement
bitstream is part of these tests. Socket fixture success is software evidence
only. Fair resident-memory Faiss comparison belongs to the transport agent's
`hw/host_benchmark` / `hw/faiss_compare` work and is not claimed here.

## Workstation Validation (2026-11-12)

GNU C++ 11.4 built the DSO and adapter compile target with
`-Wall -Wextra -Werror`. The adapter used actual executor and loader headers from
the read-only source-build tree. The runtime factory overlay passed
`patch --dry-run` against that tree. CTest passed 1/1; its Python fixture passed
all eight cases in 0.33 seconds: ABI2/success/fragmented frames, explicit config
and mock rejection, missing broker, malformed/stale/oversized/token/error/EOF
completions, deadline, cooperative cancellation and busy rejection, shutdown,
invalid object and pre-cancellation. Timeout and cancellation require zero
published results and observe socket EOF; fixture deadlines are bounded.

The PR submodule status is `-c712a30e040f19a01c511da16ba6ffd22cbe9083 sys`.
`git ls-remote https://github.com/yu-zou/ceph.git HEAD` timed out after 20 seconds;
the pinned object lookup also timed out after 120 seconds. Consequently no
clean Ceph fixture or actual OSD startup was run. No claim of hardware execution,
residual correctness, resident DDR performance or end-to-end RGW validation is
supported by this software fixture. Required follow-up is a clean ABI2 Ceph
checkout plus the transport agent's reviewed CPF1 broker, bounded device I/O,
exclusive ownership and supported arithmetic profile. Unsupported historical A6
profiles must return `-ENOTSUP`.
