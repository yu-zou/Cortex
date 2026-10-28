# ==============================================================
# Task 11: Generate Bitstream for DaisyPlus_M.2 PCIe MIG Design
# Target: XCZU17EG-FFVC1760-2-E
# Tool: Vivado 2025.2
# ==============================================================
# Fixed version: addresses floating_point sub-IP synthesis failure
# and DDR4 MIG regeneration.
# ==============================================================

# ----- Configuration -----
set project_path "D:/RAG/DaisyPlus_M.2_PCIe_MIG_201901_20210414/ps.xpr"
set output_dir "D:/RAG/GitHubPrj/Cortex/hw"
set num_jobs 4

# ----- Open Project -----
puts "============================================"
puts "Task 11: Bitstream Generation"
puts "Vivado Version: [version -short]"
puts "============================================"

open_project $project_path

# ----- Step 0: Generate output products for floating_point sub-IPs -----
puts "\n--- Step 0: Generating output products for acc_top sub-IPs ---"
set fp_ips [list \
    acc_top_fadd_32ns_32ns_32_8_full_dsp_1_ip \
    acc_top_faddfsub_32ns_32ns_32_8_full_dsp_1_ip \
    acc_top_fmul_32ns_32ns_32_5_max_dsp_1_ip \
    acc_top_fsub_32ns_32ns_32_8_full_dsp_1_ip \
]

foreach ip_name $fp_ips {
    set ip_obj [get_ips -quiet $ip_name]
    if {[llength $ip_obj] > 0} {
        puts "  Generating targets for $ip_name..."
        generate_target all $ip_obj
    } else {
        puts "  WARNING: IP '$ip_name' not found in project"
    }
}

# Note: DDR4 MIG is a nested sub-design within the block design
# It will be regenerated as part of the BD generate_target and synth_1 run.

# Generate all BD output products (this handles nested IPs like DDR4)
puts "  Generating block design output products..."
generate_target all [get_files design_1.bd]

# Export IP user files (only for standalone IPs, not BD-managed ones)
puts "  Exporting IP user files..."
export_ip_user_files -of_objects [get_ips -quiet $fp_ips] -no_script -sync -force

# ----- Step 1: Reset all runs -----
puts "\n--- Step 1: Resetting all runs ---"
catch { reset_run synth_1 }
catch { reset_run impl_1 }

# Reset individual IP synth runs for the floating_point IPs
foreach ip_name $fp_ips {
    set synth_run_name "design_1_${ip_name}_0_synth_1"
    catch { reset_run $synth_run_name }
}
catch { reset_run design_1_ddr4_0_0_synth_1 }
catch { reset_run design_1_acc_top_0_0_synth_1 }

# ----- Step 2: Synthesis -----
puts "\n--- Step 2: Launching Synthesis (synth_1) with $num_jobs jobs ---"
set synth_start [clock seconds]
launch_runs synth_1 -jobs $num_jobs
wait_on_run synth_1
set synth_end [clock seconds]
set synth_elapsed [expr {($synth_end - $synth_start) / 60}]
puts "Synthesis completed in ${synth_elapsed} minutes"

# Check synthesis status
set synth_progress [get_property PROGRESS [get_runs synth_1]]
set synth_status [get_property STATUS [get_runs synth_1]]
puts "Synthesis progress: $synth_progress"
puts "Synthesis status: $synth_status"

if {$synth_progress ne "100%"} {
    puts "ERROR: Synthesis failed with status: $synth_status"
    # Write error evidence
    set fp_out [open "$output_dir/bitstream_task11.log" w]
    puts $fp_out "Task 11: Bitstream Generation - FAILED"
    puts $fp_out "Synthesis: FAIL"
    puts $fp_out "Synthesis Status: $synth_status"
    puts $fp_out "Synthesis Progress: $synth_progress"
    close $fp_out
    exit 1
}

puts "Synthesis: PASS"

# ----- Step 3: Implementation -----
puts "\n--- Step 3: Launching Implementation (impl_1) with $num_jobs jobs ---"
set impl_start [clock seconds]
launch_runs impl_1 -jobs $num_jobs
wait_on_run impl_1
set impl_end [clock seconds]
set impl_elapsed [expr {($impl_end - $impl_start) / 60}]
puts "Implementation completed in ${impl_elapsed} minutes"

# Check implementation status
set impl_progress [get_property PROGRESS [get_runs impl_1]]
set impl_status [get_property STATUS [get_runs impl_1]]
puts "Implementation progress: $impl_progress"
puts "Implementation status: $impl_status"

if {$impl_progress ne "100%"} {
    puts "ERROR: Implementation failed with status: $impl_status"
    set fp_out [open "$output_dir/bitstream_task11.log" w]
    puts $fp_out "Task 11: Bitstream Generation - FAILED"
    puts $fp_out "Synthesis: PASS (${synth_elapsed} min)"
    puts $fp_out "Implementation: FAIL"
    puts $fp_out "Implementation Status: $impl_status"
    close $fp_out
    exit 1
}

puts "Implementation: PASS"

# ----- Step 4: Generate Bitstream -----
puts "\n--- Step 4: Generating Bitstream ---"
set bit_start [clock seconds]
launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
wait_on_run impl_1
set bit_end [clock seconds]
set bit_elapsed [expr {($bit_end - $bit_start) / 60}]
puts "Bitstream generation completed in ${bit_elapsed} minutes"

# Check bitstream status
set bit_progress [get_property PROGRESS [get_runs impl_1]]
if {$bit_progress ne "100%"} {
    puts "ERROR: Bitstream generation failed"
    exit 1
}

puts "Bitstream: PASS"

# ----- Step 5: Generate Reports -----
puts "\n--- Step 5: Generating Reports ---"
open_run impl_1

# Utilization report
set util_file "$output_dir/utilization_report.txt"
report_utilization -file $util_file
puts "Utilization report written to: $util_file"

# Timing report
set timing_file "$output_dir/timing_report.txt"
report_timing_summary -file $timing_file
puts "Timing report written to: $timing_file"

# ----- Step 6: Copy Bitstream -----
puts "\n--- Step 6: Copying Bitstream ---"
set bit_src "[file dirname $project_path]/ps.runs/impl_1/design_1_wrapper.bit"
set bit_dst "$output_dir/design_1_wrapper.bit"

if {[file exists $bit_src]} {
    file copy -force $bit_src $bit_dst
    set bit_size [file size $bit_dst]
    puts "Bitstream copied to: $bit_dst (${bit_size} bytes)"
} else {
    puts "ERROR: Bitstream file not found at: $bit_src"
    exit 1
}

# ----- Step 7: Write Summary Log -----
puts "\n--- Step 7: Writing Summary ---"

# Read timing results
set wns "N/A"
set whs "N/A"
set wpws "N/A"
if {[file exists $timing_file]} {
    set tf [open $timing_file r]
    set timing_content [read $tf]
    close $tf
    
    # Parse WNS
    if {[regexp -- {WNS\(ns\)\s*:\s*([-\d.]+)} $timing_content match wns_val]} {
        set wns $wns_val
    }
    # Parse WHS
    if {[regexp -- {WHS\(ns\)\s*:\s*([-\d.]+)} $timing_content match whs_val]} {
        set whs $whs_val
    }
    # Parse WPWS
    if {[regexp -- {WPWS\(ns\)\s*:\s*([-\d.]+)} $timing_content match wpws_val]} {
        set wpws $wpws_val
    }
}

# Read utilization results
set lut_used "N/A"
set ff_used "N/A"
set bram_used "N/A"
set dsp_used "N/A"
if {[file exists $util_file]} {
    set uf [open $util_file r]
    set util_content [read $uf]
    close $uf
}

# Write task log
set log_file "$output_dir/bitstream_task11.log"
set lf [open $log_file w]
puts $lf "================================================================"
puts $lf "Task 11: Bitstream Generation - Evidence Log"
puts $lf "================================================================"
puts $lf "Date: [clock format [clock seconds]]"
puts $lf "Vivado Version: [version -short]"
puts $lf "Target Device: [get_property PART [current_project]]"
puts $lf ""
puts $lf "--- Run Status ---"
puts $lf "Synthesis: PASS (${synth_elapsed} min)"
puts $lf "Synthesis Status: $synth_status"
puts $lf "Implementation: PASS (${impl_elapsed} min)"
puts $lf "Implementation Status: $impl_status"
puts $lf "Bitstream: PASS (${bit_elapsed} min)"
puts $lf ""
puts $lf "--- Timing Results ---"
puts $lf "WNS (Worst Negative Slack): ${wns} ns"
puts $lf "WHS (Worst Hold Slack): ${whs} ns"
puts $lf "WPWS (Worst Pulse Width Slack): ${wpws} ns"
if {$wns ne "N/A" && $whs ne "N/A"} {
    set timing_met [expr {$wns >= 0 && $whs >= 0 ? "YES" : "NO"}]
} else {
    set timing_met "UNKNOWN"
}
puts $lf "Timing Met: $timing_met"
puts $lf ""
puts $lf "--- Bitstream Output ---"
puts $lf "Source: $bit_src"
puts $lf "Destination: $bit_dst"
puts $lf "Bitstream Size: ${bit_size} bytes"
puts $lf ""
puts $lf "--- Utilization Summary ---"
puts $lf "(See utilization_report.txt for full details)"
puts $lf ""
puts $lf "--- Files Generated ---"
puts $lf "Utilization Report: $util_file"
puts $lf "Timing Report: $timing_file"
puts $lf "Bitstream: $bit_dst"
puts $lf "================================================================"
close $lf

puts "\n============================================"
puts "Task 11 COMPLETED SUCCESSFULLY"
puts "Total time: synth=${synth_elapsed}min, impl=${impl_elapsed}min, bitstream=${bit_elapsed}min"
puts "WNS: ${wns} ns, WHS: ${whs} ns"
puts "============================================"

exit
