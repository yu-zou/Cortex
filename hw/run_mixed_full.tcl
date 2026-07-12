open_project mixed_full_proj -reset
set_top acc_top

add_files src/acc_top.cpp
add_files -tb test/testbench_mixed_full.cpp

open_solution solution1 -flow_target vivado
set_part {xcvu5p-flva2104-1-e}
create_clock -period 5 -name default

# C Simulation
csim_design -clean

# C Synthesis
csynth_design

exit
