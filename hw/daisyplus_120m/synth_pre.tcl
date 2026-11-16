# Avoid the observed Vivado 2019.1 HARTS parallelClose helper-monitor stall.
set_param synth.enableParallelHelperSpawn none
set_param synth.maxThreads 1
set_param general.maxThreads 1
puts "SYNTH_HELPER_SPAWN_DISABLED"
