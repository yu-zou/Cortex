#!/bin/bash
# ============================================================
# IVF-PQ FPGA Accelerator — Vitis HLS Build Script
# Target: Xilinx VU5P (xcvu5p-flva2104-1-e) @ 200 MHz
# ============================================================
#
# Prerequisites:
#   - Xilinx Vitis HLS 2021.2 installed at /tools/Xilinx/Vitis_HLS/2021.2
#   - Ubuntu 20.04 (or compatible)
#
# Usage:
#   ./run.sh          # Full flow: C simulation + C synthesis + RTL export
#   ./run.sh sim      # C simulation only
#   ./run.sh syn      # C synthesis + RTL export only
#   ./run.sh clean    # Remove generated files
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VITIS_DIR="/tools/Xilinx/Vitis_HLS/2021.2"

# ─── Check Vitis HLS Installation ───
if [ ! -f "${VITIS_DIR}/settings64.sh" ]; then
    echo "ERROR: Vitis HLS not found at ${VITIS_DIR}"
    echo "Please update VITIS_DIR in this script to match your installation."
    exit 1
fi

# ─── Source Vitis HLS Environment ───
echo "============================================"
echo " IVF-PQ FPGA Accelerator — Vitis HLS Build"
echo "============================================"
echo ""
echo "Sourcing Vitis HLS 2021.2 ..."
source "${VITIS_DIR}/settings64.sh"

# ─── Fix CRT path for Ubuntu 20.04 ───
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH}"

# ─── Parse mode ───
MODE="${1:-full}"

case "${MODE}" in
    sim)
        echo ""
        echo ">>> Running C Simulation only <<<"
        vitis_hls -f "${SCRIPT_DIR}/run_sim.tcl"
        ;;
    syn)
        echo ""
        echo ">>> Running C Synthesis + RTL Export <<<"
        vitis_hls -f "${SCRIPT_DIR}/run_syn.tcl"
        ;;
    clean)
        echo ""
        echo ">>> Cleaning generated files <<<"
        rm -rf "${SCRIPT_DIR}/acc_top_proj"
        rm -f  "${SCRIPT_DIR}/vitis_hls.log"
        echo "Done."
        ;;
    full|*)
        echo ""
        echo ">>> Running Full Flow: C Simulation + C Synthesis + RTL Export <<<"
        vitis_hls -f "${SCRIPT_DIR}/run_hls.tcl"
        ;;
esac

echo ""
echo "============================================"
echo " Build complete."
echo " Reports: ${SCRIPT_DIR}/acc_top_proj/solution1/syn/report/"
echo " RTL:     ${SCRIPT_DIR}/acc_top_proj/solution1/impl/"
echo "============================================"
