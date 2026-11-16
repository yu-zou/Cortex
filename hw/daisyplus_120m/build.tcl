set out [file normalize [lindex $argv 0]]
set root [file normalize [file dirname [info script]]]
if {![string match {2019.1*} [version -short]]} {error "Vivado 2019.1 required, got [version -short]"}
set_param general.maxThreads 4
source "$root/synth_pre.tcl"
set src "$out/src/daisyplus_xdma2_bd_bundle_slim/shared/daisyplus_acc_top_v2/ps_fix_v14_fixed_120m_accpblock"
set shared "$out/src/daisyplus_xdma2_bd_bundle_slim/shared"
set bd "$src/ps_fix.srcs/sources_1/bd/design_1/design_1.bd"
create_project daisyplus_120m "$out/project" -part xczu17eg-ffvc1760-2-e
set_property ip_repo_paths [list "$shared/acc_top_repack_ip"] [current_project]
update_ip_catalog
add_files -norecurse $bd
add_files -norecurse "$src/ps_fix.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v"
foreach name {constr xdma_constr nvme_constr pcie_ep_constr mig_const} {
    add_files -fileset constrs_1 -norecurse "$shared/daisyplus_acc_top_v2/ps.srcs/constrs_1/new/$name.xdc"
}
add_files -fileset constrs_1 -norecurse "$shared/fix_gt_clocks_baseline.xdc"
add_files -fileset constrs_1 -norecurse "$src/acc_no_gt_s28.xdc"
set_property used_in_synthesis false [get_files "$src/acc_no_gt_s28.xdc"]
set_property top design_1_wrapper [current_fileset]
open_bd_design [get_files $bd]
foreach name {xdma_0 xdma_1 xdma_2 ddr4_0 acc_top_0} {
    if {[llength [get_bd_cells -quiet /$name]] != 1} {error "Missing BD block $name"}
}
validate_bd_design
reset_target all [get_files $bd]
generate_target all [get_files $bd]
puts "BD_VALIDATION_AND_TARGET_GENERATION_COMPLETE"
set_property synth_checkpoint_mode Hierarchical [get_files $bd]
create_ip_run [get_files $bd]
set ip_runs [get_runs -quiet -filter {IS_SYNTHESIS == 1 && NAME != synth_1}]
if {![llength $ip_runs]} {error "No IP synthesis runs created"}
foreach run [get_runs -filter {IS_SYNTHESIS == 1}] {
    set_property STEPS.SYNTH_DESIGN.TCL.PRE "$root/synth_pre.tcl" $run
}
launch_runs $ip_runs -jobs 2
foreach ip_run $ip_runs {
    wait_on_run $ip_run
    puts "IP_SYNTH_STATUS ${ip_run}: [get_property STATUS $ip_run]"
    if {![string match {*Complete*} [get_property STATUS $ip_run]]} {
        error "IP synthesis failed: ${ip_run}: [get_property STATUS $ip_run]"
    }
}

create_run impl_baseline -parent_run synth_1 -flow {Vivado Implementation 2019}
set run [get_runs impl_baseline]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED false $run
catch {set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED false $run}
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt $run
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {![string match {*Complete!*} [get_property STATUS [get_runs synth_1]]]} {error "Synthesis failed: [get_property STATUS [get_runs synth_1]]"}
launch_runs impl_baseline -to_step place_design -jobs 4
wait_on_run impl_baseline
set status [get_property STATUS $run]
if {$status ne "Not started route_design" && ![string match {*place_design Complete*} $status]} {error "Baseline placement failed: $status"}
open_run impl_baseline
write_checkpoint -force "$out/placed_baseline.dcp"
report_timing_summary -report_unconstrained -file "$out/placement_timing.rpt"
close_design
open_checkpoint "$out/placed_baseline.dcp"
route_design -directive AggressiveExplore
write_checkpoint -force "$out/routed_baseline.dcp"
write_xdc -force "$out/baseline_constraints.xdc"

# The historical single grouping contains all three XDMA domains and fabric.
set f [open "$out/baseline_constraints.xdc" r]
set text [read $f]
close $f
set filtered {}
set removed 0
foreach line [split $text \n] {
    if {[string match {set_clock_groups*} $line] && [string first {*xdma_0*GTHE4_CHANNEL_PRIM_INST/TXOUTCLK*} $line] >= 0} {
        incr removed
    } else {append filtered $line \n}
}
if {$removed != 1} {error "Expected exactly one baseline grouping, removed $removed"}
set f [open "$out/repaired_constraints.xdc" w]
puts -nonewline $f $filtered
close $f
reset_timing
read_xdc "$out/repaired_constraints.xdc"
read_xdc "$shared/fix_gt_clocks.xdc"
route_design -unroute
route_design -directive Explore
write_checkpoint -force "$out/routed.dcp"
report_timing_summary -report_unconstrained -file "$out/timing.rpt"
report_route_status -file "$out/route_status.rpt"
report_bus_skew -file "$out/bus_skew.rpt"
report_drc -file "$out/drc.rpt"
report_cdc -file "$out/cdc.rpt"
check_timing -verbose -file "$out/check_timing.rpt"
report_clocks -file "$out/clocks.rpt"
report_exceptions -file "$out/exceptions.rpt"
set acc_clock [get_clocks -of_objects [get_pins design_1_i/acc_top_0/ap_clk]]
if {[llength $acc_clock] != 1 || abs([get_property PERIOD $acc_clock] - 8.333333) > 0.002} {
    error "Accelerator clock is missing, ambiguous, or not 120 MHz"
}

set from [get_cells -hier -filter {NAME =~ *xdma_2*512b_rq_intfc_mod*s_axis_rq_tdata_reg*}]
set to [get_cells -hier -filter {NAME =~ *xdma_2*512b_rq_intfc_mod*write_data_reg*}]
if {![llength $from] || ![llength $to]} {error "XDMA2 conversion endpoints not found"}
foreach delay {min max} {
    report_timing -from $from -to $to -delay_type $delay -max_paths 20 -file "$out/conversion_$delay.rpt"
    set p [get_timing_paths -from $from -to $to -delay_type $delay -max_paths 1]
    if {[llength $p] != 1 || [get_property SLACK $p] < 0 || [string match -nocase *inf* [get_property SLACK $p]]} {error "XDMA2 $delay timing missing, unconstrained or failing"}
    set p [get_timing_paths -delay_type $delay -max_paths 1]
    if {[llength $p] != 1 || [get_property SLACK $p] < 0 || [string match -nocase *inf* [get_property SLACK $p]]} {error "Global $delay timing missing or failing"}
}
if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error}]]} {error "DRC errors present"}
source "$root/gates.tcl"
check_reports $out
write_bitstream -force "$out/design_1_wrapper.bit"
puts "BUILD_COMPLETE_REVIEW_REPORTS"
exit
