open_project acc_top_proj -reset
set_top compute_engine

add_files src/acc_top.cpp
add_files -tb tb/testbench.cpp

open_solution solution1 -flow_target vivado
set_part {xcvu5p-flva2104-1-e}
create_clock -period 5 -name default

# C Simulation only
csim_design -clean

exit
