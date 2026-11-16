# Run with tclsh; no Xilinx tools or hardware are used.
source [file join [file dirname [info script]] gates.tcl]
set route {# of routable nets..................... : 10 :
# of fully routed nets................. : 10 :
# of nets with routing errors.......... : 0 :}
set coverage {There are 0 register/latch pins with no clock.
There are 0 pins that are not constrained for maximum delay.}
set skew {Slack (MET) : 0.010ns}
check_report_text $route $coverage $skew
foreach {name r c s} [list \
    unrouted [string map {{fully routed nets................. : 10} {fully routed nets................. : 9}} $route] $coverage $skew \
    route_errors [string map {{: 0 :} {: 1 :}} $route] $coverage $skew \
    no_clock $route [string map {{0 register} {12 register}} $coverage] $skew \
    unconstrained $route [string map {{0 pins} {20 pins}} $coverage] $skew \
    missing_coverage $route {} $skew \
    failing_skew $route $coverage {Slack (VIOLATED) : -0.01ns} \
    missing_skew $route $coverage {} \
    missing_route {} $coverage $skew] {
    if {![catch {check_report_text $r $c $s} message]} {error "Gate accepted $name"}
    puts "PASS reject $name: $message"
}
foreach path {build.tcl gates.tcl test_gates.tcl} {
    set f [open [file join [file dirname [info script]] $path]]
    set text [read $f]
    close $f
    if {![info complete $text]} {error "Incomplete Tcl: $path"}
}
puts "STATIC_TCL_GATES_PASS"
if {[llength $argv]} {
    set release [lindex $argv 0]
    foreach {name filename} {route route_status.rpt coverage timing.rpt skew bus_skew.rpt} {
        set f [open [file join $release $filename]]
        set $name [read $f]
        close $f
    }
    if {![catch {check_report_text $route $coverage $skew} message]} {
        error "Historical incomplete-clock candidate unexpectedly passed"
    }
    if {![string match {Timing coverage failed:*} $message]} {error "Unexpected historical rejection: $message"}
    puts "PASS historical candidate rejected: $message"
}
