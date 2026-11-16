# Safe Host Control Contract

No device access is authorized for this implementation. `VfioTransport` and
direct positional XDMA construction fail before opening anything. The original
VFIO test's descriptor layout, AXI addresses and historical register sequence
are retained in `host.py`; its pagemap IOVA selection, PCI bus-master write,
implicit process cleanup and direct VFIO opening are not safe defaults.

`OwnerTransport` is an actual bounded socket transport, not a CPU search. It
borrows a connected Unix stream from an embedding service. It neither opens nor
closes device descriptors. It requires a separately implemented and reviewed
persistent device owner. **No compatible hardware owner is supplied or verified
here, so hardware execution remains blocked.** A socket reply alone is not
hardware attestation; its kind is deliberately `owner-rpc-unverified`.

## Wire Protocol

Version 1 request and reply headers are 48 little-endian bytes, Python format
`<8sIIQQQiI`: magic `CXOWN001`, version=1, operation (1=write, 2=read),
monotonically increasing request ID, AXI address, byte length, signed status,
reserved zero. Write requests append exactly length bytes. Read replies append
exactly length bytes. Successful replies echo every header field, status=0.
Negative errno status, malformed headers, disconnects, short transfers and
timeouts quarantine the client connection. Maximum payload is 1 MiB. No retry.

The owner must authenticate its local peer, independently enforce the same
address allowlist and exclusive lease, and reject other clients while leased.
Only accelerator DDR windows and the enumerated historical AXI control
registers are allowed. PCI configuration, raw SSD LBAs, reset and rebinding have
no operations. Each acknowledgement means DMA finished with exact descriptor
count and no error bits, and host/device visibility is established. A submitted
DMA descriptor retains its mapping until verified quiescence. The owner needs
its own finite hardware deadlines; client deadlines do not cancel DMA.

`begin`/`end` serialize transactions across sessions sharing one transport.
The embedding application must also serialize staging and searches across the
entire resident index lifecycle; restaging while another session runs is not
allowed. Cancellation stops polling/publication and quarantines the transport.
Timeout, error or disconnect must never release the owner's VFIO FDs or mappings.
The owner retains these resources and rejects later work until an authorized
operator establishes quiescence. No automatic reset, unmap, close or CPU fallback.

## VFIO Release Hazard

Omitting `VFIO_DEVICE_RESET` does not make VFIO safe: the kernel may reset on
first open or last device-FD release, including process death. Borrowing an FD
does not by itself fix that. This implementation resolves the client hazard by
never acquiring a VFIO FD; it does not claim that an absent owner resolves the
kernel hazard. Enabling direct VFIO requires evidence for the exact running
kernel/driver reset scope, exclusive ownership of every affected function and
an approved open/release policy. A persistent service alone is insufficient
proof because it can crash. Until that evidence and authorization exist, fail
closed, including on the current bitstream whose generated map is unverified.

## Layout and ABI2

`layout.py` defines `CXIVFPQ1`, a 128-byte little-endian list envelope:
magic[8], eleven uint32 fields (version, header bytes, list ID, flags, D, M,
nbits, count, centroid bytes, codebook bytes, record bytes), SHA256[32], zero[44].
Flags: bit0 residual L2, bit1 IDs are document addresses; other bits rejected.
Payload is centroid[768] FP32, codebook[96][256][8] FP32, count records of 128
bytes. All floats must be finite. Header lengths, hash and unique IDs are checked.
Empty lists are represented by the benchmark offsets, never launched.

Each record has codes 0..63 reversed in bytes 0..63, zero padding 64..83,
uint32 document length at 84, uint64 ID/address at 88, codes 64..95 reversed
at 96..127. This follows actual `adc_engine` bit slices. The 64-byte raw device
header contains M, D, count as uint32 then zeros. Historical A6 keeps codebook
separate and emits 64-byte result slots. Current source embeds the codebook after
the header and emits packed 16-byte results; its control map is not assumed.

Inspected workstation `include/cortex_driver_api.h` ABI2 has eight semantic-read
arguments, document addresses and 24-byte aligned host entries. The PR checkout
still has ABI1. Neither is the FPGA packed result ABI. OSD integration must decode
`<fQI` then assign host fields and zero padding, check ABI2 before calling, and
require `decode(..., require_document_addresses=True)`. Faiss benchmark exports
carry vector IDs and cannot bless OSD reads. No OSD/driver files are modified.
