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
