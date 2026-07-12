open_project unified_proj -reset
set_top acc_top

add_files src/acc_top.cpp
add_files -tb test/testbench_unified.cpp

open_solution solution1 -flow_target vivado
set_part {xcvu5p-flva2104-1-e}
create_clock -period 5 -name default

csim_design -clean

exit
