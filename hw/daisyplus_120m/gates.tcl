# Fail closed if Vivado's report format differs from the audited format.
proc check_reports {out} {
    foreach name {route_status check_timing bus_skew} {
        set f [open "$out/$name.rpt" r]
        set report($name) [read $f]
        close $f
    }
    check_report_text $report(route_status) $report(check_timing) $report(bus_skew)
}

proc check_report_text {route coverage skew} {
    foreach {label key} {{routable nets} routable {fully routed nets} routed {nets with routing errors} errors} {
        set pattern [format {# of %s\.+ *: *([0-9]+)} $label]
        if {![regexp $pattern $route -> count($key)]} {
            error "Missing route metric: $label"
        }
    }
    if {$count(routable) == 0 || $count(routable) != $count(routed) || $count(errors) != 0} {
        error "Routing incomplete or has errors"
    }
    foreach pattern {{There are ([0-9]+) register/latch pins with no clock} {There are ([0-9]+) pins that are not constrained for maximum delay\.}} {
        set matches [regexp -all -inline $pattern $coverage]
        if {![llength $matches]} {error "Missing timing coverage metric: $pattern"}
        foreach {text endpoints} $matches {
            if {$endpoints != 0} {error "Timing coverage failed: $text"}
        }
    }
    if {![regexp {Slack \(MET\)} $skew] || [regexp {VIOLATED} $skew]} {
        error "Bus-skew checks missing or failing"
    }
}
