open_project acc_top_proj
set_top acc_top

add_files src/acc_top.cpp
add_files -tb test/testbench.cpp

open_solution solution1 -flow_target vivado
set_part {xczu17eg-ffvc1760-2-e}
create_clock -period 5 -name default

# C Synthesis
csynth_design

# Export RTL
export_design -format ip_catalog -vendor "ivfpq" -version "1.0"

exit
