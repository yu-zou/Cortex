#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_IMAGE="ceph-build:main.ubuntu22.04"
TEST_IMAGE="cortex-integration-test"

# --- Base image availability check ---
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    echo "Base image $BASE_IMAGE not found locally. Building it..."
    pushd "$PROJECT_ROOT/sys" >/dev/null
    python3 src/script/build-with-container.py -d ubuntu22.04 -b build.u2204 -e build-container
    popd >/dev/null
    if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
        echo "ERROR: Failed to build base image $BASE_IMAGE" >&2
        exit 1
    fi
    echo "Base image $BASE_IMAGE built successfully."
else
    echo "Base image $BASE_IMAGE found locally."
fi

# --- Parse optional flags ---
NO_CACHE=""
KEEP=""
REBUILD_BASE=""
for arg in "$@"; do
    case "$arg" in
        --no-cache)    NO_CACHE="--no-cache" ;;
        --keep)        KEEP=1 ;;
        --rebuild-base)
            echo "Rebuilding base image $BASE_IMAGE..."
            pushd "$PROJECT_ROOT/sys" >/dev/null
            python3 src/script/build-with-container.py -d ubuntu22.04 -b build.u2204 -e build-container
            popd >/dev/null
            ;;
    esac
done

# --- Build test image ---
echo "Building $TEST_IMAGE from Cortex/ root..."
docker build --network=host $NO_CACHE \
    -f "$PROJECT_ROOT/test/integration/Dockerfile" \
    -t "$TEST_IMAGE" \
    "$PROJECT_ROOT"

# --- Run test (includes Ceph cluster bootstrap + FPGA verification + tests) ---
echo "Running integration test (Ceph cluster + FPGA verification + tests)..."
START_TIME=$(date +%s)

if [ -n "$KEEP" ]; then
    docker run --name cortex-integ-run "$TEST_IMAGE"
    RUN_EXIT=$?
else
    docker run --rm "$TEST_IMAGE"
    RUN_EXIT=$?
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ "$RUN_EXIT" -eq 0 ]; then
    echo "Integration test PASSED (${ELAPSED}s)"
else
    echo "Integration test FAILED (exit=$RUN_EXIT, ${ELAPSED}s)" >&2
fi

exit "$RUN_EXIT"
