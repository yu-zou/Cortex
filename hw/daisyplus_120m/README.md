# DaisyPlus XDMA2/A6 120 MHz FPGA package

This package reconstructs the source project from the pinned release archive
and invokes Vivado 2019.1 with one command. It does not include generated
netlists, checkpoints, implementation runs, bitstreams, boot components, or a
Vivado installation.

## Build

Requirements: Bash, Python 3.9+, `tar` with zstd support, `sha256sum`, and Vivado 2019.1
with XCZU17EG device support and the required IP/implementation licenses.

```sh
bash hw/daisyplus_120m/build.sh
```

`VIVADO` may name a non-default Vivado executable and `BUILD_DIR` may select a
new absolute or relative output directory. Existing output directories are
never overwritten. The wrapper verifies the complete archive SHA-256 before
extracting only the active block design/wrapper, custom `acc_top` IP
repository, board constraints, and clock-fix constraints. Vivado-generated
files and all build products stay in the new output directory.
From inside this directory, use `bash build.sh`. For extraction only, use
`bash build.sh prepare`; that mode never invokes Vivado. Preparation does not
validate the BD or IP catalog and also requires a fresh output directory.

For this workstation's isolated licensed runtime, run one command from the
checkout root:

```sh
bash hw/daisyplus_120m/run_container.sh
```

This prepares and audits sources with host Python before launching a detached
Ubuntu 16.04 container. Docker mounts the copied Vivado installation, compatible
runtime libraries and license files read-only; license contents are not logged.
The network is disabled, resources limited to 8 CPUs/40 GiB, and timeout is 24h.
`BUILD_DIR` must be a fresh path within this package. The container is retained
after exit for diagnosis. Use `docker logs daisyplus-lean-licensed` for progress.
The launcher refuses an active isolated build and an existing output directory.
`VIVADO_TOOL_ROOT`, `VIVADO_RUNTIME_ROOT`, and `VIVADO_LICENSE_DIR` can override
the workstation dependency locations documented in `run_container.sh`.

The controller and every synthesis run source `synth_pre.tcl`, disabling
parallel helper spawning and limiting synthesis to one thread. This targets the
observed 2019.1 HARTS helper shutdown hang; implementation remains a real tool
run, with all report gates retained.

The flow performs project and BD checks, IP target generation, synthesis,
baseline placement and routing, then replaces the baseline clock grouping and
reroutes with the corrected constraint. The old grouping is used only for the
historically required initial place/route stage. A bitstream is emitted only
after explicit global and XDMA2 conversion setup/hold checks, no unconstrained
internal endpoints or unclocked sequential pins, no route errors, no bus-skew
failures, and no DRC errors. Reports and logs remain in the build directory.

These gates are not full signoff. Review every warning, clock and exception,
CDC report, route status, unconstrained path report, and timing report before
using an artifact. The previously board-tested 120 MHz implementation reported
small timing margin (global setup +0.002 ns, hold +0.006 ns); changed source or
tool versions require fresh review. No BOOT.BIN or Linux image is created.

## Source and external dependencies

`SOURCE_ARCHIVE` defaults to
`./daisyplus_xdma2_bd_bundle_slim.tar.zst`; set it to a local
copy of that archive if needed. Required SHA-256:

```
c7a3342b32bb051e88496ac68d53d3e8d7f729dc73cc86ec4f6f6d2d7bc11d5f
```

The archive is an external source dependency, not vendored source. Vivado
2019.1's proprietary IP catalogs, device database, licensed primitives, and
implementation tools are also external. The selected extraction is declared
in `sources.list`. No generated hardware netlist or checkpoint is substituted
for these source dependencies.

The selection includes the filtered `ps_fix.srcs` tree (BD, wrappers, vendor
RTL/XCI/data), five board XDCs, the placement
XDC, baseline and corrected clock XDCs, and the `acc_top` 1.4 IP repository.
The latter includes synthesizable HLS-generated RTL, initialization `.dat`
files, four floating-point XCI subcores, IP-XACT metadata, XGUI Tcl, and the
OOC XDC. Vendor-generated BD/IP output products are regenerated. This does not
rebuild HLS from C++, and does not establish equivalence to `hw/src/acc_top.cpp`
at the PR head. No public archive download URL was established here; users
must obtain an identical copy separately.

The pinned slim archive is 366,110,554 bytes (about 349 MiB). Its top-level
directory is `daisyplus_xdma2_bd_bundle_slim/`. `sources.list` paths are relative
to that directory and remain unchanged: selection excludes the archive's boot
images, candidate bitstreams and old work directories. Preparation also excludes
DCPs, simulation netlists and HWDEF files; it retains RTL/XCI/data needed for
regeneration. Vivado must regenerate missing output targets.

The slim bundle's `open_bd.tcl` and this repository's `build.tcl` register the
same BD, wrapper, custom IP repository, five board XDCs and accelerator Pblock.
Their scope differs: the bundle entry opens the BD and skips simulation
references in its dependency check; this repository uses the baseline grouping
for initial placement/routing, then the final grouping, with stricter report
gates before bitgen. The archive's own `build.sh` is not invoked. Successful
slim BD opening does not validate this full build. Known clock-model/OOC and
implementation issues remain; switching archives does not resolve timing closure.

## Verification status and known risks

2026-11-16 slim archive integration: `build.sh prepare` passed with 1,183
files, 184,900,154 bytes and 104 resolved custom-IP references. The unchanged
relative entries in `sources.list` all exist in the slim archive. Shell syntax
and the Tcl report-gate fixtures passed; no Vivado build was run for this change.
There are still differences requiring a separate build repair/validation: the
archive custom-IP OOC XDC specifies 5.556 ns although the BD requests 120 MHz;
the final archive grouping only matches the unsuffixed XDMA2 user clock; and
the repository flow still uses `open_run` after placement. These are not fixed
by selecting a different archive and this integration is not a timing pass.

The source package is not represented as a verified clean full build. The
previous clean-build attempt regenerated BD/IP targets but emitted BD
critical/warning diagnostics, including reset of BRAM initialization, address
mapping, tied-off pins, and width warnings. Its full licensed IP synthesis was
observed stalled; GDB confirmed a HARTS helper shutdown join wait and the exact
isolated container was gracefully stopped under user authorization. A
separate earlier source flow timed out during Vivado project creation. Neither
constitutes a successful synthesis, implementation, bitstream, or board test.

Historical timing experiments show why a simple one-pass route is unsafe: the
corrected clock grouping exposes a related 250-to-500 MHz XDMA2 RQ conversion
path that an old broad asynchronous grouping masked. Historical three-stage
results are not results for this package's clean build. The Tcl gates enforce
selected checks but cannot prove constraint intent, CDC correctness, full
exception coverage, or hardware operation. Do not use a generated bitstream
until all reports and board-level validation have been reviewed.

The archive's historical three-stage candidate actually reports 20,466
unclocked register/latch pins and 42,737 unconstrained internal endpoints in
`candidates/a6_bundle_slim_final_20261106/timing.rpt`. Its positive slack is
not complete timing signoff. The new coverage gate rejects those counts.
Missing reference-clock models and any resulting setup/hold failures remain
an implementation blocker. No guessed clock frequency or false-path waiver
has been added to make a report pass.

Other clean-build pitfalls: Vivado 2019.1 requires compatible Linux runtime
libraries (earlier attempts encountered missing libX11/runtime failures),
correct device/IP licenses, sufficient RAM/disk and long-running IP synthesis.
IP synthesis runs are explicitly created and awaited before top synthesis.
BRAM reset warnings require review: the prior original XCI specified no
COE/MEM initialization, but this does not prove memory behavior is correct.
BD warnings are retained in the log. The baseline XDC uses Tcl commands that
can cause XDC warnings. Replacement requires exactly one matching exported
grouping and fails if the export format differs. Report gates also fail on
unknown report formats. A licensed full build and board validation are still
required.

## Static verification

On 2026-11-12 the expanded filtered archive preparation and `check_sources.py`
passed: 1,313 files, 192,143,502 bytes (about 183 MiB), 104 resolved IP-XACT file
references, four parseable XCI subcores, required BD cells and requested
120 MHz clock verified. No DCP, EDIF, bitstream, hardware definition or
simulation netlist was selected. The checked-in files are only scripts,
manifest and documentation; prepared assets are ignored build inputs.

`bash -n build.sh`, Tcl completeness checks and report-gate tests passed.
The tests accept a passing fixture and reject incomplete routing, route
errors, missing/nonzero coverage, missing/failing bus skew and absent route
metrics. They also reject the historical candidate for its real unclocked
pins. These are static tests, not execution of Vivado commands. Run:

```sh
tclsh hw/daisyplus_120m/test_gates.tcl
BUILD_DIR=/path/to/new/output bash hw/daisyplus_120m/build.sh prepare
```

An earlier detached Vivado 2019.1 build used `lean-build-helper-off`.
See `DIAGNOSIS.md` for active process evidence and subsequent results. Initial
controller logs confirm the helper workaround was accepted. Full synthesis,
route, timing and bitstream are not yet validated. No hardware access, commit
or push was performed. `source.sha256` records inputs before regeneration;
`scripts.sha256` records the launched script versions.
