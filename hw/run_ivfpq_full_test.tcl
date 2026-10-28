open_project ivfpq_full_proj -reset
set_top compute_engine

add_files src/acc_top.cpp
add_files -tb test/testbench_ivfpq_full.cpp

open_solution solution1 -flow_target vivado
set_part {xcvu5p-flva2104-1-e}
create_clock -period 5 -name default

# C Simulation only (full pipeline: data_manager + compute_engine)
csim_design -clean

exit
csynth_design
exit
