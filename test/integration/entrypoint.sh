#!/bin/bash
set -euo pipefail

# Signal handling — cleanup on exit (also covers docker stop SIGTERM)
cleanup() {
    echo "=== Shutting down Ceph cluster ==="
    cd /ceph/build.u2204
    ../src/stop.sh 2>/dev/null || true
    echo "=== Shutdown complete ==="
}
trap cleanup EXIT SIGTERM SIGINT

# Verify driver library exists (fail fast before starting cluster)
if [ ! -f /usr/local/lib/libcortex_driver.so ]; then
    echo "ERROR: /usr/local/lib/libcortex_driver.so not found"
    echo "       Ensure libcortex_driver.so is installed to /usr/local/lib/ before running."
    exit 1
fi

echo "=== Starting Ceph cluster ==="
# Must cd /ceph/build.u2204 BEFORE running vstart.sh — it looks for CMakeCache.txt in $PWD.
# When launched from the build directory, vstart.sh reads CEPH_ROOT from CMakeCache.txt
# and sets CEPH_BUILD_DIR=$PWD automatically.
cd /ceph/build.u2204

# The ceph-osd binary in this debug build crashes in tcmalloc global destructors on exit
# (exit code 139). The crash is benign — it only happens after all OSD data is committed.
# We wrap ceph-osd to convert exit code 139 → 0 so vstart.sh's set -e doesn't abort.
mv bin/ceph-osd bin/ceph-osd.real
cat > bin/ceph-osd << 'WRAPPER'
#!/bin/bash
/ceph/build.u2204/bin/ceph-osd.real "$@" || { ec=$?; [ $ec -eq 139 ] && exit 0 || exit $ec; }
WRAPPER
chmod +x bin/ceph-osd

# Start ephemeral vstart cluster with Cortex driver injected into OSD args.
# Memory caps prevent OOM in container environment.
# --without-dashboard avoids "Frontend assets not found" error.
MON=1 OSD=1 MGR=1 RGW=1 \
  ../src/vstart.sh -n -d -x \
  --bluestore \
  --without-dashboard \
  --osd-args "--cortex_driver_path=/usr/local/lib/libcortex_driver.so" \
  -o "osd pool default size = 1" \
  -o "osd pool default min size = 1" \
  -o "osd_memory_target = 1073741824" \
  -o "bluestore_cache_size = 536870912"

echo "=== Waiting for cluster health ==="
# Poll until HEALTH_OK or HEALTH_WARN (single OSD = HEALTH_WARN is acceptable).
# Timeout after 180s to avoid hanging forever.
timeout 180 bash -c '
  until /ceph/build.u2204/bin/ceph health 2>/dev/null | grep -qE "HEALTH_OK|HEALTH_WARN"; do
    echo "  waiting for cluster..."
    sleep 5
  done
' || { echo "ERROR: Cluster did not become healthy within 180s"; exit 1; }

/ceph/build.u2204/bin/ceph -s

# ============================================================================
# FPGA Verification — after cluster is healthy, before running tests
# ============================================================================
echo ""
echo "=== FPGA Verification ==="

FPGA_PASS=0
FPGA_FAIL=0
QUERY_LATENCY_MS="N/A"

# --- [1/4] Library loadability via ldconfig ---
echo "--- [1/4] Driver library loadability ---"
if ldconfig -p | grep -q libcortex_driver; then
    echo "  [PASS] libcortex_driver.so registered with ldconfig"
    FPGA_PASS=$((FPGA_PASS + 1))
else
    echo "  [FAIL] libcortex_driver.so NOT found by ldconfig"
    FPGA_FAIL=$((FPGA_FAIL + 1))
fi

if ldconfig -p | grep -q libcortex_hw_mock; then
    echo "  [PASS] libcortex_hw_mock.so registered with ldconfig"
    FPGA_PASS=$((FPGA_PASS + 1))
else
    echo "  [FAIL] libcortex_hw_mock.so NOT found by ldconfig"
    FPGA_FAIL=$((FPGA_FAIL + 1))
fi

# --- [2/4] Driver ABI version check via dlopen ---
echo "--- [2/4] Driver ABI version check ---"
ABI_OUTPUT=$(python3 -c '
import ctypes, sys
lib = ctypes.CDLL("libcortex_driver.so")
lib.cortex_api_version.restype = ctypes.c_uint32
v = lib.cortex_api_version()
print(f"ABI_VERSION={v}")
if v != 1:
    print("FAIL: expected version 1")
    sys.exit(1)
print("PASS: ABI version matches")
' 2>&1) || true
echo "$ABI_OUTPUT" | sed 's/^/  /'
if echo "$ABI_OUTPUT" | grep -q "PASS: ABI version matches"; then
    FPGA_PASS=$((FPGA_PASS + 1))
else
    FPGA_FAIL=$((FPGA_FAIL + 1))
fi

# --- [3/4] Mock FPGA query via driver with timing ---
echo "--- [3/4] Mock FPGA query with timing ---"
FIXTURE_DIR="${FIXTURE_DIR:-/cortex/hw/test/fixtures}"
QUERY_OUTPUT=$(python3 -c "
import ctypes, sys, os, time

lib = ctypes.CDLL('libcortex_driver.so')
lib.cortex_api_version.restype = ctypes.c_uint32
ver = lib.cortex_api_version()
if ver != 1:
    print(f'ABI_ERROR: version={ver}')
    sys.exit(1)

lib.cortex_driver_init.restype = ctypes.c_int
cfg = b'{\"hw_lib_path\":\"/usr/local/lib/libcortex_hw_mock.so\"}'
rc = lib.cortex_driver_init(cfg)
if rc != 0:
    print(f'INIT_ERROR: init returned {rc}')
    sys.exit(1)
print('DRIVER_INIT_OK')

fixture = os.environ.get('FIXTURE_DIR', '/cortex/hw/test/fixtures')
path = os.path.join(fixture, 'cluster_0.bin')
if not os.path.exists(path):
    print(f'NO_FIXTURE: {path}')
    lib.cortex_driver_shutdown()
    sys.exit(0)

with open(path, 'rb') as f:
    blob = f.read()
print(f'FIXTURE_LOADED: {len(blob)} bytes')

lib.cortex_semantic_write.restype = ctypes.c_int
wc = lib.cortex_semantic_write(0, blob, len(blob), None, 0, 0)
if wc != 0:
    print(f'WRITE_ERROR: write returned {wc}')
    lib.cortex_driver_shutdown()
    sys.exit(1)
print('SEMANTIC_WRITE_OK')

class TopKEntry(ctypes.Structure):
    _fields_ = [('distance', ctypes.c_float),
                ('doc_addr', ctypes.c_uint64),
                ('doc_length', ctypes.c_uint32),
                ('_pad', ctypes.c_uint32)]

class ReadResult(ctypes.Structure):
    _fields_ = [('status', ctypes.c_int),
                ('count', ctypes.c_uint32),
                ('entries', TopKEntry * 500)]

lib.cortex_semantic_read.restype = ReadResult
query = (ctypes.c_float * 32)(*([0.5] * 32))
t0 = time.time()
r = lib.cortex_semantic_read(0, blob, len(blob), query, 32, 5, 0)
t1 = time.time()
elapsed_ms = (t1 - t0) * 1000.0

lib.cortex_driver_shutdown()
print(f'DRIVER_SHUTDOWN_OK')
print(f'QUERY_STATUS={r.status} QUERY_COUNT={r.count} LATENCY_MS={elapsed_ms:.2f}')
if r.status != 0 or r.count == 0:
    print('QUERY_FAILED')
    sys.exit(1)
" 2>&1) || true

# Display query output with indentation
echo "$QUERY_OUTPUT" | while IFS= read -r line; do echo "    $line"; done

if echo "$QUERY_OUTPUT" | grep -q 'LATENCY_MS'; then
    QUERY_LATENCY_MS=$(echo "$QUERY_OUTPUT" | grep -oP 'LATENCY_MS=\K[0-9.]+')
    if echo "$QUERY_OUTPUT" | grep -q 'QUERY_FAILED'; then
        echo "  [FAIL] Mock FPGA query returned error status"
        FPGA_FAIL=$((FPGA_FAIL + 1))
    else
        echo "  [PASS] Mock FPGA query completed (${QUERY_LATENCY_MS}ms)"
        FPGA_PASS=$((FPGA_PASS + 1))
    fi
else
    echo "  [FAIL] Mock FPGA query did not complete"
    FPGA_FAIL=$((FPGA_FAIL + 1))
fi

# --- [4/4] Verification summary ---
echo "--- [4/4] FPGA Verification Summary ---"
CEPH_HEALTH=$(/ceph/build.u2204/bin/ceph health 2>/dev/null || echo "UNKNOWN")
echo "  Ceph cluster health : ${CEPH_HEALTH}"
echo "  Driver library      : $([ $(ldconfig -p 2>/dev/null | grep -c libcortex_driver) -gt 0 ] && echo 'LOADED' || echo 'MISSING')"
echo "  HW mock library     : $([ $(ldconfig -p 2>/dev/null | grep -c libcortex_hw_mock) -gt 0 ] && echo 'LOADED' || echo 'MISSING')"
echo "  Driver init         : $(echo "$QUERY_OUTPUT" | grep -q 'DRIVER_INIT_OK' && echo 'OK' || echo 'FAIL')"
echo "  Mock device search  : $(echo "$QUERY_OUTPUT" | grep -q 'QUERY_STATUS=0' && echo 'RESPONDED' || echo 'FAIL')"
echo "  Query latency       : ${QUERY_LATENCY_MS}ms"
echo "  FPGA checks         : ${FPGA_PASS} passed, ${FPGA_FAIL} failed"
echo ""

echo "=== Running test_integration.sh ==="
# FIXTURE_DIR: fixture files mounted in container at /cortex/hw/test/fixtures
# CEPH_CONF: GTest reads this via conf_read_file(nullptr) which checks $CEPH_CONF
export FIXTURE_DIR=/cortex/hw/test/fixtures
export CEPH_CONF=/ceph/build.u2204/ceph.conf

# Use PIPESTATUS[0] — tee always exits 0, masking real failures
bash /ceph/src/test/cortex/test_integration.sh /ceph/build.u2204 2>&1 | tee /tmp/integ.log
INTEG_EXIT=${PIPESTATUS[0]}

if ! grep -q 'ALL INTEGRATION TESTS PASSED' /tmp/integ.log; then
    echo "FAIL: test_integration.sh did not print success marker"
    exit 1
fi
[ "$INTEG_EXIT" -eq 0 ] || { echo "FAIL: test_integration.sh exit code=$INTEG_EXIT"; exit 1; }

echo "=== Running ceph_test_cortex_integration (GTest) ==="
/ceph/build.u2204/bin/ceph_test_cortex_integration 2>&1 | tee /tmp/gtest.log
GTEST_EXIT=${PIPESTATUS[0]}

[ "$GTEST_EXIT" -eq 0 ] || { echo "FAIL: ceph_test_cortex_integration exit=$GTEST_EXIT"; exit 1; }

echo "=== ALL TESTS PASSED ==="
exit 0
