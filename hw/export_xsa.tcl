# Task 9: Export XSA with bitstream
# Open the project
open_project D:/RAG/DaisyPlus_M.2_PCIe_MIG_201901_20210414/ps.xpr

# Open block design
open_bd_design [get_files design_1.bd]

# Generate output products for design_1.bd
generate_target all [get_files design_1.bd]
save_bd_design

# Export hardware with bitstream
write_hw_platform -fixed -include_bit -force -file D:/RAG/GitHubPrj/Cortex/hw/acc_top.xsa

puts "XSA exported successfully to D:/RAG/GitHubPrj/Cortex/hw/acc_top.xsa"
close_project
