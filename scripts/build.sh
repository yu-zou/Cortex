#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
LIB_DIR="${BUILD_DIR}/lib"

PROXY="http://192.168.104.167:7897"
HLS_SETTINGS="${VITIS_HLS_SETTINGS:-/tools/Xilinx/Vitis_HLS/2021.2/settings64.sh}"

mkdir -p "$LIB_DIR"

MODE="${1:-build}"

# ── Helper ────────────────────────────────────────────────────────
section() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "══════════════════════════════════════════════════════════════"
}

# ── Mode dispatch ─────────────────────────────────────────────────
case "${MODE}" in

    # ═══════════════════════════════════════════════════════════════
    #  BUILD  (original behavior, preserved)
    # ═══════════════════════════════════════════════════════════════
    build)
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
        ;;

    # ═══════════════════════════════════════════════════════════════
    #  TEST-HW  — build hw/ + run CTest
    # ═══════════════════════════════════════════════════════════════
    test-hw)
        section "TEST SUITE: hw/ CTest"
        docker build --network=host \
          --build-arg http_proxy="$PROXY" \
          --build-arg https_proxy="$PROXY" \
          -t cortex-hw-build \
          -f "${PROJECT_ROOT}/hw/Dockerfile" \
          "${PROJECT_ROOT}"

        docker run --rm --network=host \
          -v "${PROJECT_ROOT}:/cortex" \
          -v "${LIB_DIR}:/output" \
          cortex-hw-build bash -c '
            set -euo pipefail
            mkdir -p /build && cd /build
            cmake /cortex/hw -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/output
            make -j"$(nproc)"
            make install
            echo "--- Running hw/ CTest ---"
            ctest --output-on-failure
            echo "HW_CTEST_PASSED"
        '
        ;;

    # ═══════════════════════════════════════════════════════════════
    #  TEST-DRIVER  — build hw/ + driver/ + run driver CTest
    # ═══════════════════════════════════════════════════════════════
    test-driver)
        section "TEST SUITE: driver/ CTest"

        # 1) Build hw image and stage hw libraries (driver depends on them)
        docker build --network=host \
          --build-arg http_proxy="$PROXY" \
          --build-arg https_proxy="$PROXY" \
          -t cortex-hw-build \
          -f "${PROJECT_ROOT}/hw/Dockerfile" \
          "${PROJECT_ROOT}"

        docker run --rm --network=host \
          -v "${PROJECT_ROOT}:/cortex" \
          -v "${LIB_DIR}:/output" \
          cortex-hw-build bash -c '
            set -euo pipefail
            mkdir -p /build && cd /build
            cmake /cortex/hw -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/output
            make -j"$(nproc)"
            make install
        '

        # 2) Build driver image + run driver CTest
        docker build --network=host \
          --build-arg http_proxy="$PROXY" \
          --build-arg https_proxy="$PROXY" \
          -t cortex-driver-build \
          -f "${PROJECT_ROOT}/driver/Dockerfile" \
          "${PROJECT_ROOT}"

        docker run --rm --network=host \
          -v "${PROJECT_ROOT}:/cortex" \
          -v "${LIB_DIR}:/output" \
          cortex-driver-build bash -c '
            set -euo pipefail
            mkdir -p /build && cd /build
            cmake /cortex/driver -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX=/output \
              -DCORTEX_HW_LIB_DIR=/output/lib
            make -j"$(nproc)"
            make install
            echo "--- Running driver/ CTest ---"
            ctest --output-on-failure
            echo "DRIVER_CTEST_PASSED"
        '
        ;;

    # ═══════════════════════════════════════════════════════════════
    #  TEST-HLS  — Vitis HLS C simulation via run_*_test.tcl scripts
    # ═══════════════════════════════════════════════════════════════
    test-hls)
        section "TEST SUITE: Vitis HLS C Simulation"

        if [ ! -f "$HLS_SETTINGS" ]; then
            echo "ERROR: Vitis HLS not found at ${HLS_SETTINGS}"
            echo "       Set VITIS_HLS_SETTINGS env var to the settings64.sh path."
            exit 1
        fi

        # Source settings with errexit safety
        set +e
        # shellcheck disable=SC1090
        source "$HLS_SETTINGS"
        RC=$?
        set -e
        if [ "$RC" -ne 0 ]; then
            echo "ERROR: Failed to source Vitis HLS settings: ${HLS_SETTINGS}"
            exit 1
        fi

        PASS=0
        FAIL=0
        FAILED_TESTS=""

        cd "${PROJECT_ROOT}/hw"

        for tcl_script in run_*_test.tcl; do
            [ -f "$tcl_script" ] || continue
            echo ""
            echo "--- HLS Test: ${tcl_script} ---"
            if vitis_hls -f "$tcl_script" 2>&1; then
                echo "--- ${tcl_script}: PASS ---"
                PASS=$((PASS + 1))
            else
                echo "--- ${tcl_script}: FAIL ---"
                FAIL=$((FAIL + 1))
                FAILED_TESTS="${FAILED_TESTS}  ${tcl_script}"$'\n'
            fi
        done

        echo ""
        section "HLS Test Summary: ${PASS} passed, ${FAIL} failed"
        if [ "$FAIL" -gt 0 ]; then
            echo "Failed tests:"
            echo -n "$FAILED_TESTS"
            exit 1
        fi
        echo "HLS_TESTS_PASSED"
        ;;

    # ═══════════════════════════════════════════════════════════════
    #  TEST-E2E  — build integration Docker image + run
    # ═══════════════════════════════════════════════════════════════
    test-e2e)
        section "TEST SUITE: E2E Integration"
        echo "(Building integration Docker image—this may take a while)"
        docker build --network=host \
          --build-arg HTTP_PROXY="$PROXY" \
          --build-arg HTTPS_PROXY="$PROXY" \
          -t cortex-e2e-test \
          -f "${PROJECT_ROOT}/test/integration/Dockerfile" \
          "${PROJECT_ROOT}"

        section "Running E2E Integration Tests"
        docker run --rm --network=host \
          cortex-e2e-test
        echo "E2E_TESTS_PASSED"
        ;;

    # ═══════════════════════════════════════════════════════════════
    #  TEST  — run ALL test suites sequentially
    # ═══════════════════════════════════════════════════════════════
    test)
        section "TEST MODE: Running all test suites"
        PASS=0
        FAIL=0

        echo ""
        echo "───── test-hw ─────"
        "$0" test-hw && PASS=$((PASS + 1)) || FAIL=$((FAIL + 1))

        echo ""
        echo "───── test-driver ─────"
        "$0" test-driver && PASS=$((PASS + 1)) || FAIL=$((FAIL + 1))

        echo ""
        echo "───── test-hls ─────"
        "$0" test-hls && PASS=$((PASS + 1)) || FAIL=$((FAIL + 1))

        echo ""
        echo "───── test-e2e ─────"
        "$0" test-e2e && PASS=$((PASS + 1)) || FAIL=$((FAIL + 1))

        echo ""
        section "ALL TESTS: ${PASS} passed, ${FAIL} failed"
        [ "$FAIL" -eq 0 ] || exit 1
        echo "ALL_TESTS_PASSED"
        ;;

    # ═══════════════════════════════════════════════════════════════
    #  UNKNOWN MODE
    # ═══════════════════════════════════════════════════════════════
    *)
        echo "Usage: $0 {build|test|test-hw|test-driver|test-hls|test-e2e}"
        echo ""
        echo "  build       Default. Build hw/ + driver/ artifacts."
        echo "  test        Build + run ALL test suites (hw CTest, driver CTest, HLS sim, E2E)."
        echo "  test-hw     Build + run hw/ CTest targets only."
        echo "  test-driver Build + run driver/ CTest targets only."
        echo "  test-hls    Run Vitis HLS C simulation testbenches."
        echo "  test-e2e    Build + run E2E pipeline integration test."
        exit 1
        ;;

esac
