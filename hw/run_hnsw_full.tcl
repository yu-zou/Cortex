# ═══════════════════════════════════════════════════════════════
# run_hnsw_full.tcl — Full HNSW Pipeline Test
# Tests hnsw_search_engine end-to-end with DRAM preload path.
#
# Usage:
#   vitis -s run_hnsw_full.tcl
#   # or in vitis_hls shell:
#   source run_hnsw_full.tcl
# ═══════════════════════════════════════════════════════════════

open_project hnsw_full_proj -reset
set_top hnsw_search_engine

# Source files
add_files src/acc_top.cpp

# Testbench
add_files -tb test/testbench_hnsw_full.cpp

open_solution solution1 -flow_target vivado
set_part {xcvu5p-flva2104-1-e}
create_clock -period 5 -name default

# C Simulation (functional verification)
csim_design -clean

# C Synthesis (RTL generation)
csynth_design

exit
