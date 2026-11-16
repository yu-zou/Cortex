"""Version 1 OSD list envelope, separate from device and C function ABIs."""
import hashlib
import math
import struct

MAGIC = b'CXIVFPQ1'
# magic, version, header_bytes, list_id, flags, D, M, nbits, count,
# centroid_bytes, codebook_bytes, record_bytes, payload_sha256, reserved
HEADER = struct.Struct('<8s11I32s44s')
VERSION = 1
RESIDUAL = 1
ID_IS_DOCUMENT_ADDRESS = 2


def encode(list_id, cluster, codebook, centroid, *, residual, document_addresses=False):
    import host
    n = host.validate_inputs(cluster, codebook, bytes(3072), 1)
    if len(centroid) != 3072 or not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', centroid)):
        raise ValueError('finite FP32 centroid required')
    if not 0 <= list_id <= 0xffffffff or any(cluster[12:64]):
        raise ValueError('invalid list ID or reserved cluster bytes')
    records = cluster[64:]
    ids = [struct.unpack_from('<Q', records, i*128+88)[0] for i in range(n)]
    if len(set(ids)) != n:
        raise ValueError('duplicate IDs')
    payload = centroid+codebook+records
    flags = int(residual) | (ID_IS_DOCUMENT_ADDRESS if document_addresses else 0)
    return HEADER.pack(MAGIC, VERSION, HEADER.size, list_id, flags, 768, 96, 8, n,
                       3072, 786432, 128, hashlib.sha256(payload).digest(), bytes(44))+payload


def decode(blob, *, require_document_addresses=False):
    if len(blob) < HEADER.size:
        raise ValueError('short envelope')
    magic, version, size, lid, flags, dim, m, bits, n, cq, cb, stride, digest, reserved = HEADER.unpack_from(blob)
    if (magic, version, size, dim, m, bits, cq, cb, stride) != (MAGIC, VERSION, HEADER.size, 768, 96, 8, 3072, 786432, 128):
        raise ValueError('unsupported envelope layout')
    if flags & ~3 or any(reserved) or not 1 <= n <= 8192:
        raise ValueError('invalid flags, reserved bytes or count')
    if require_document_addresses and not flags & ID_IS_DOCUMENT_ADDRESS:
        raise ValueError('benchmark vector IDs are not OSD document addresses')
    payload = blob[size:]
    if len(payload) != cq+cb+n*stride or hashlib.sha256(payload).digest() != digest:
        raise ValueError('envelope length/hash mismatch')
    centroid, codebook = payload[:cq], payload[cq:cq+cb]
    cluster = struct.pack('<III', m, dim, n)+bytes(52)+payload[cq+cb:]
    # Validate the same contract in both directions.
    encode(lid, cluster, codebook, centroid, residual=bool(flags & RESIDUAL),
           document_addresses=bool(flags & ID_IS_DOCUMENT_ADDRESS))
    return dict(list_id=lid, residual=bool(flags & RESIDUAL),
                document_addresses=bool(flags & ID_IS_DOCUMENT_ADDRESS),
                centroid=centroid, codebook=codebook, cluster=cluster)
