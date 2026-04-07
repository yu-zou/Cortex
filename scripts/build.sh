#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
LIB_DIR="${BUILD_DIR}/lib"

PROXY="http://192.168.104.167:7897"

mkdir -p "$LIB_DIR"

echo "=== Building hw/ (mock SmartSSD device) ==="
docker build --network=host \
  --build-arg http_proxy="$PROXY" \
  --build-arg https_proxy="$PROXY" \
  -t cortex-hw-build \
  -f "${PROJECT_ROOT}/hw/Dockerfile" \
  "${PROJECT_ROOT}"

docker run --rm --network=host \
  -v "${PROJECT_ROOT}:/cortex" \
  -v "${LIB_DIR}:/output" \
  cortex-hw-build bash -c \
    "mkdir -p /build && cd /build && \
     cmake /cortex/hw -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/output && \
     make -j\$(nproc) && \
     make install && \
     ls -la /output/lib/libcortex_hw_mock.so && \
     echo HW_BUILD_OK"

echo "=== Building driver/ (host driver .so) ==="
docker build --network=host \
  --build-arg http_proxy="$PROXY" \
  --build-arg https_proxy="$PROXY" \
  -t cortex-driver-build \
  -f "${PROJECT_ROOT}/driver/Dockerfile" \
  "${PROJECT_ROOT}"

docker run --rm --network=host \
  -v "${PROJECT_ROOT}:/cortex" \
  -v "${LIB_DIR}:/output" \
  cortex-driver-build bash -c \
    "mkdir -p /build && cd /build && \
     cmake /cortex/driver -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/output -DCORTEX_HW_LIB_DIR=/output/lib && \
     make -j\$(nproc) && \
     make install && \
     ls -la /output/lib/libcortex_driver.so && \
     echo DRIVER_BUILD_OK"

echo "=== Artifacts staged in ${LIB_DIR} ==="
ls -la "$LIB_DIR"
echo "ALL_BUILDS_OK"

echo ""
echo "=== Integration Test Instructions ==="
echo "To run the end-to-end integration test inside the Ceph build container:"
echo ""
echo "  docker run --rm --network=host \\"
echo "    -v ${PROJECT_ROOT}:/cortex \\"
echo "    -e http_proxy=http://192.168.104.167:7897 \\"
echo "    -e https_proxy=http://192.168.104.167:7897 \\"
echo "    ceph-build:main.ubuntu22.04 bash -c '"
echo "      cp /cortex/build/lib/libcortex_driver.so /usr/local/lib/ 2>/dev/null || true"
echo "      cp /cortex/build/lib/libcortex_hw_mock.so /usr/local/lib/ 2>/dev/null || true"
echo "      ldconfig"
echo "      cd /cortex/sys"
echo "      ninja -C build.u2204 semantic_op_tool 2>/dev/null"
echo "      MON=1 OSD=3 RGW=1 /ceph/src/vstart.sh -n -d -x --bluestore --without-dashboard \\"
echo "        --osd-args \"--cortex_driver_path=/usr/local/lib/libcortex_driver.so\" 2>&1 | tail -20"
echo "      sleep 15"
echo "      build.u2204/bin/ceph -s"
echo "      bash src/test/cortex/test_integration.sh build.u2204 2>&1 | tee /tmp/integ.log"
echo "      grep -q 'ALL INTEGRATION TESTS PASSED' /tmp/integ.log && echo INTEG_OK"
echo "    '"
echo ""
echo "Note: Requires ceph-build:main.ubuntu22.04 image and a completed sys/build.u2204 build."
