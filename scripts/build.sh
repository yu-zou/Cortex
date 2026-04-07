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
