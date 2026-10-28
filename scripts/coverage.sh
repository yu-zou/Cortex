#!/bin/bash
# ============================================================================
# coverage.sh — Test coverage report generation (gcov / lcov)
#
# Measures line coverage for hw/src/ and driver/src/ directories.
# Generates an HTML report via lcov + genhtml for easy visual inspection.
#
# Usage:
#   ./scripts/coverage.sh            # full coverage cycle
#   ./scripts/coverage.sh --open     # open HTML report after generation
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_DIR="${PROJECT_ROOT}/build_coverage"
COV_LIB_DIR="${BUILD_DIR}/lib"
COV_REPORT_DIR="${PROJECT_ROOT}/coverage_report"
COV_INFO="${BUILD_DIR}/coverage.info"

OPEN_BROWSER=false
for arg in "$@"; do
    case "$arg" in
        --open) OPEN_BROWSER=true ;;
        -h|--help)
            echo "Usage: $0 [--open]"
            echo ""
            echo "  --open    Open HTML coverage report in browser after generation"
            exit 0
            ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
section() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "══════════════════════════════════════════════════════════════"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

warn() {
    echo "WARNING: $*" >&2
}

# ── Tool availability checks ────────────────────────────────────────────────
check_tools() {
    local missing=false

    if ! command -v gcov &>/dev/null; then
        warn "gcov not found. Install it with:  sudo apt install gcc  (or g++-<ver>)"
        missing=true
    fi

    if ! command -v lcov &>/dev/null; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║  lcov not found — install it:                                       ║"
        echo "║                                                                      ║"
        echo "║    sudo apt install lcov                                             ║"
        echo "║                                                                      ║"
        echo "║  If you only want to run tests with coverage instrumentation and      ║"
        echo "║  skip report generation, re-run with:                                 ║"
        echo "║    COVERAGE_SKIP_LCOV=1 $0                                          ║"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
        echo ""
        if [[ -z "${COVERAGE_SKIP_LCOV:-}" ]]; then
            die "lcov is required for report generation. Install it or set COVERAGE_SKIP_LCOV=1 to skip."
        fi
    fi

    if ! command -v genhtml &>/dev/null; then
        if [[ -z "${COVERAGE_SKIP_LCOV:-}" ]]; then
            die "genhtml not found (part of lcov package). Install:  sudo apt install lcov"
        fi
    fi

    if ! command -v cmake &>/dev/null; then
        die "cmake not found. Install it first."
    fi

    if ! command -v make &>/dev/null; then
        die "make not found. Install build-essential first."
    fi

    if ! command -v ctest &>/dev/null; then
        warn "ctest not found — tests won't be run automatically."
    fi

    if "$missing"; then
        die "Required tools missing. Install and re-run."
    fi
}

# ── Step 0: Check existing build directory ─────────────────────────────────
check_build_dir() {
    if [[ -d "$BUILD_DIR" ]]; then
        echo "WARNING: Coverage build directory already exists at:"
        echo "  ${BUILD_DIR}"
        echo ""
        echo "Options:"
        echo "  1) Remove and recreate (fresh build with coverage flags)"
        echo "  2) Keep and resume (assumes coverage-flavored build)"
        echo "  3) Abort"
        echo ""
        echo -n "Choose [1/recreate, 2/keep, 3/abort]: "
        read -r choice
        case "${choice:-1}" in
            1|recreate)
                echo "Removing ${BUILD_DIR} ..."
                rm -rf "$BUILD_DIR"
                mkdir -p "$BUILD_DIR"
                echo "Fresh build directory created."
                ;;
            2|keep)
                echo "Using existing build directory."
                ;;
            3|abort)
                die "Aborted by user."
                ;;
            *)
                die "Invalid choice. Aborting."
                ;;
        esac
    else
        mkdir -p "$BUILD_DIR"
    fi
}

# ── Step 1–2: Configure and build hw/ with coverage flags ──────────────────
build_hw() {
    section "Building hw/ with coverage flags"

    local HW_BUILD_DIR="${BUILD_DIR}/hw"
    mkdir -p "$HW_BUILD_DIR"

    cmake -S "${PROJECT_ROOT}/hw" -B "$HW_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="--coverage -O0 -g -fno-inline -fno-elide-constructors" \
        -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
        -DCMAKE_SHARED_LINKER_FLAGS="--coverage" \
        -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}"

    cmake --build "$HW_BUILD_DIR" -j "$(nproc)"
    cmake --install "$HW_BUILD_DIR" --prefix "${BUILD_DIR}"

    echo "hw/ build complete. Library staged at ${COV_LIB_DIR}"
}

# ── Step 1–2: Configure and build driver/ with coverage flags ──────────────
build_driver() {
    section "Building driver/ with coverage flags"

    local DRV_BUILD_DIR="${BUILD_DIR}/driver"
    mkdir -p "$DRV_BUILD_DIR"

    cmake -S "${PROJECT_ROOT}/driver" -B "$DRV_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="--coverage -O0 -g -fno-inline -fno-elide-constructors" \
        -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
        -DCMAKE_SHARED_LINKER_FLAGS="--coverage" \
        -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
        -DCORTEX_HW_LIB_DIR="${COV_LIB_DIR}"

    cmake --build "$DRV_BUILD_DIR" -j "$(nproc)"
    cmake --install "$DRV_BUILD_DIR" --prefix "${BUILD_DIR}"

    echo "driver/ build complete."
}

# ── Step 3: Run CTest for both projects ────────────────────────────────────
run_tests() {
    section "Running CTest — hw/ tests"

    local HW_BUILD_DIR="${BUILD_DIR}/hw"
    if [[ -d "$HW_BUILD_DIR" ]]; then
        # Note: hw tests use CORTEX_FIXTURE_DIR defined in CMakeLists.txt
        # (resolved via ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/).
        (cd "$HW_BUILD_DIR" && ctest --output-on-failure) || \
            warn "Some hw/ tests FAILED (see above). Coverage data may be incomplete."
    else
        warn "hw/ build directory not found — skipping hw CTest."
    fi

    section "Running CTest — driver/ tests"
    local DRV_BUILD_DIR="${BUILD_DIR}/driver"
    if [[ -d "$DRV_BUILD_DIR" ]]; then
        (cd "$DRV_BUILD_DIR" && ctest --output-on-failure) || \
            warn "Some driver/ tests FAILED (see above). Coverage data may be incomplete."
    else
        warn "driver/ build directory not found — skipping driver CTest."
    fi
}

# ── Step 4: lcov capture ───────────────────────────────────────────────────
capture_coverage() {
    section "Capturing coverage data with lcov"

    # Initialize tracefile
    lcov --capture --initial --directory "${BUILD_DIR}" \
        --output-file "${COV_INFO}.base" \
        --rc lcov_branch_coverage=1 2>/dev/null

    # Capture after test execution
    lcov --capture --directory "${BUILD_DIR}" \
        --output-file "${COV_INFO}.test" \
        --rc lcov_branch_coverage=1 2>/dev/null

    # Merge base + test
    lcov --add-tracefile "${COV_INFO}.base" \
        --add-tracefile "${COV_INFO}.test" \
        --output-file "${COV_INFO}" \
        --rc lcov_branch_coverage=1 2>/dev/null

    # Extract only hw/src/ and driver/src/
    lcov --extract "${COV_INFO}" \
        "${PROJECT_ROOT}/hw/src/*" \
        "${PROJECT_ROOT}/driver/src/*" \
        --output-file "${COV_INFO}.filtered" \
        --rc lcov_branch_coverage=1 2>/dev/null

    mv "${COV_INFO}.filtered" "${COV_INFO}"

    # Remove intermediate files
    rm -f "${COV_INFO}.base" "${COV_INFO}.test"

    echo "Coverage data captured to ${COV_INFO}"
}

# ── Step 5: Generate HTML report ──────────────────────────────────────────
generate_html() {
    section "Generating HTML coverage report"

    mkdir -p "$COV_REPORT_DIR"

    genhtml "${COV_INFO}" \
        --output-directory "$COV_REPORT_DIR" \
        --rc lcov_branch_coverage=1 \
        --title "Cortex Coverage Report" \
        --legend \
        --highlight \
        --function-coverage \
        --branch-coverage 2>/dev/null

    echo ""
    echo "HTML report generated at:"
    echo "  file://${COV_REPORT_DIR}/index.html"
}

# ── Step 6: Print summary ──────────────────────────────────────────────────
print_summary() {
    section "Coverage Summary"

    if [[ ! -f "$COV_INFO" ]]; then
        warn "No coverage data found at ${COV_INFO}"
        return
    fi

    echo ""
    echo "── Overall line coverage ──────────────────────────────"
    lcov --summary "${COV_INFO}" --rc lcov_branch_coverage=1 2>&1 | \
        grep -E '^(lines|functions|branches)\.*:' || true

    echo ""
    echo "── Per-file coverage (hw/src/) ────────────────────────"
    lcov --list "${COV_INFO}" --rc lcov_branch_coverage=1 2>&1 | \
        grep -E "(hw/src/|Total:)" || \
        warn "No hw/src/ files in coverage data."

    echo ""
    echo "── Per-file coverage (driver/src/) ────────────────────"
    lcov --list "${COV_INFO}" --rc lcov_branch_coverage=1 2>&1 | \
        grep -E "(driver/src/|Total:)" || \
        warn "No driver/src/ files in coverage data."

    # Extract overall line coverage percentage
    local line_pct
    line_pct=$(lcov --summary "${COV_INFO}" --rc lcov_branch_coverage=1 2>&1 | \
        grep 'lines\.*:' | awk '{print $2}' | tr -d '%')
    if [[ -n "$line_pct" ]]; then
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        printf "  OVERALL LINE COVERAGE:  %.1f%%\n" "$line_pct"
        echo "══════════════════════════════════════════════════════════════"
    fi
}

# ── Open HTML in browser ───────────────────────────────────────────────────
open_report() {
    local index="${COV_REPORT_DIR}/index.html"
    if [[ ! -f "$index" ]]; then
        warn "Report not found at ${index} — cannot open."
        return
    fi

    echo ""
    echo "Opening coverage report in browser..."

    if command -v xdg-open &>/dev/null; then
        xdg-open "$index" 2>/dev/null &
    elif command -v open &>/dev/null; then
        open "$index" 2>/dev/null &
    elif command -v gnome-open &>/dev/null; then
        gnome-open "$index" 2>/dev/null &
    elif command -v sensible-browser &>/dev/null; then
        sensible-browser "$index" 2>/dev/null &
    else
        warn "No browser opener found. Open manually:"
        echo "  file://${index}"
    fi
}

# ── Cleanup coverage artifacts ─────────────────────────────────────────────
cleanup_gcda() {
    # Remove stale .gcda files so they don't pollute a fresh run.
    find "${BUILD_DIR}" -name "*.gcda" -delete 2>/dev/null || true
}

# ── Main ────────────────────────────────────────────────────────────────────
main() {
    echo "Cortex Coverage Report Generator"
    echo "================================="
    echo "Project root : ${PROJECT_ROOT}"
    echo "Build dir    : ${BUILD_DIR}"
    echo "Report dir   : ${COV_REPORT_DIR}"
    echo ""

    check_tools
    check_build_dir
    cleanup_gcda

    build_hw
    build_driver

    run_tests

    if command -v lcov &>/dev/null && command -v genhtml &>/dev/null; then
        capture_coverage
        generate_html
        print_summary

        if "$OPEN_BROWSER"; then
            open_report
        fi
    else
        if command -v lcov &>/dev/null; then
            # lcov available but genhtml missing — still print summary
            capture_coverage
            print_summary
        fi
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  lcov/genhtml not installed — install with:"
        echo "    sudo apt install lcov"
        echo "  Then re-run to generate the HTML report."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi

    echo ""
    echo "DONE. Coverage report: file://${COV_REPORT_DIR}/index.html"
}

main
