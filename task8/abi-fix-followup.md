# ABI migration blocker follow-up — bounded evidence

Source: `/mnt/d/RAG/GitHubPrj/Cortex`.
Parent evidence actually read:
`/home/user/OpenCodePrj/.omo/evidence/cortex-plan/task-8/build-audit.md`,
especially sections 3.3 and 5. The earlier relative-path ambiguity is resolved.
Task 8 remains incomplete.

## Implemented corrections

1. `sys/src/cortex/semantic_result_guard.h` contains executable production
   helpers used by OSD/RGW. Driver status is returned as an error (positive
   invalid statuses normalize to `-EIO`); count >500 returns `-EOVERFLOW` before
   OSD indexes the fixed array. No-dequeue starts with `-EIO`, not success.
2. RGW records submission, completion, operation, decoded status, excessive
   result size and decode/incompatible-wire errors. Every submitted completion
   is drained/released, then any selected-cluster failure clears output and
   returns whole-query HTTP500 before merge. No unmarked partial response.
   Missing centroid data or no selected cluster now returns HTTP503 instead of
   empty HTTP200. No centroid population, parser, scoring or cache fixes added.
3. Semantic result/entry decoders call the shared wire-version policy helper;
   versions below 2 reject the old vector-ID schema. Ceph's DECODE_START still
   owns framing, truncation and future compatibility checks.
4. Both independent Python probes in `test/integration/entrypoint.sh` expect
   ABI2. Its final gate returns exit1 when FPGA_FAIL is nonzero, before the
   ALL TESTS PASSED marker.
5. Loader resolves into a local `API pending`, invokes init using that table,
   and publishes only on successful init. Missing-symbol/version/init/path
   failures from an unloaded state expose no API. `unload()` clears an already
   unloaded table too. An already-loaded instance rejects reload with `-EBUSY`
   and retains its live table/handle. Publication is transactional under the
   existing serialized-use assumption, not a claim of thread-safe atomicity.

## Red before loader fix

Added an ABI2 DSO missing the final required symbol
`cortex_get_last_cluster_id` and the regression before changing the loader:

```sh
cmake -S /tmp/opencode/cortex-task8-audit -B /tmp/opencode/cortex-task8-audit/build
cmake --build /tmp/opencode/cortex-task8-audit/build --target test_loader_abi -j2
/tmp/opencode/cortex-task8-audit/build/bin/test_loader_abi --gtest_filter=LoaderABI.MissingLastSymbolNeverPublishesPointers --gtest_output=xml:/tmp/opencode/cortex-task8-audit/abi-followup-red.xml
```

Observed exit1, one failed case, 12 failed pointer assertions: six exported
members remained non-null after dlclose, and remained non-null after unload.
The missing-symbol diagnostic named `cortex_get_last_cluster_id`. No stale
pointer was called. XML is retained at the absolute path above.

## Green execution

```sh
cmake -S /tmp/opencode/cortex-task8-audit -B /tmp/opencode/cortex-task8-audit/build
cmake --build /tmp/opencode/cortex-task8-audit/build -j2
env LD_LIBRARY_PATH=/home/user/miniconda3/lib ctest --test-dir /tmp/opencode/cortex-task8-audit/build --output-on-failure --output-junit /tmp/opencode/cortex-task8-audit/abi-followup-all.xml
bash -n /mnt/d/RAG/GitHubPrj/Cortex/test/integration/entrypoint.sh
/tmp/opencode/cortex-task8-audit/build/bin/test_loader_abi --gtest_output=xml:/tmp/opencode/cortex-task8-audit/abi-followup-green.xml
/tmp/opencode/cortex-task8-audit/build/bin/test_semantic_result_guard --gtest_output=xml:/tmp/opencode/cortex-task8-audit/abi-followup-guard.xml
ctest --test-dir /tmp/opencode/cortex-task8-audit/build -R '^(test_loader_abi|test_semantic_result_guard|test_entrypoint_contract)$' -V --output-log /tmp/opencode/cortex-task8-audit/abi-followup-focused.log
```

Observed **15/15 CTest targets passed**, shell syntax exit0. Last full run took
0.59s. Includes all 12 preexisting hw/driver targets, loader ABI, result guard,
and entrypoint fragment execution. The existing ADC target now passes with the
checkout's prior residual correction; this follow-up did not edit ADC sources.

- Loader: 6 GTest cases, 0 failures. Includes incomplete ABI2 DSO rejection,
  all-seven-members null, repeated unload, successful recovery, failed init,
  old ABI rejection before init, loaded reload rejection, nondefault eighth
  argument and canaries, actual driver's uninitialized export call. Incomplete
  DSO init aborts if called, proving resolution rejects before init.
- Guard: 5 GTest cases, 0 failures. Tests real production helper functions:
  wire versions 0/1 rejected and 2 accepted; count 0/500 accepted and 501/
  UINT32_MAX rejected; negative status propagation and invalid-positive status;
  all selected completions required; twelve partial/all-failed combinations
  covering submission/transport/operation/status/decode-exception/oversize;
  transport failure bypasses decode and successful later completion cannot
  erase the failure. Decode exceptions are injected callbacks, not fabricated
  Ceph wire decoding; the shared wire-version predicate is separately executed.
- Entrypoint: Python executes the actual first ABI probe fragment against real
  ABI2 driver and version1 DSO via local symlink/search path, checking success/
  rejection. Executes the actual final shell tail with FPGA_FAIL=0,1,3 and checks
  exit codes and absence/presence of success marker. This is executable fragment
  testing, not source-text assertions presented as behavioral proof. It never
  starts a cluster or executes the entire integration entrypoint.

The entrypoint fragment test was also invoked directly and printed
`entrypoint ABI 2/1 probes and final FPGA_FAIL 0/1/3 gate passed`.
LSP diagnostics were attempted for the helper, but configured `clangd` is not
installed; no LSP clean result is claimed. Actual helper compilation passed.

## Exact unresolved full-Ceph gates

No claim is made that unbuilt RGW/OSD compiled or that HTTP/wire behavior ran.
The helper tests validate policy functions, not Ceph buffers, AIO, HTTP status
serialization, or linkage. Full verification must still:

Local read checks in this follow-up reconfirmed absent `sys/src/rgw/`, absent
`sys/src/test/cortex/`, and empty `sys/src/googletest/`. `sys/src/utf8proc/` is
currently absent locally (the parent audit describes its selected remote tree
as empty). No remote inventory was repeated.

1. Restore the authenticated fork's missing `sys/src/rgw/` tree (including
   rgw_rest.h / rgw_op.h) and selected vendored sources. Parent inventory reports
   empty utf8proc, Jerasure/gf-complete, googletest; Arrow inventory also needs
   completion under the selected options. Existing Boost outputs are absent.
2. Establish measured isolated full-build source/dependency/object/link budget.
   Parent's 66.86GiB remote free-space measurement is not a build-fit proof.
3. Generate a usable Ceph build graph/config headers. Parent dry-run found no
   build.ninja; no full configure/build was launched in this follow-up.
4. Compile/link `cortex_rgw`, `radosgw`, `ceph-osd`, specifically changed
   `rgw_semantic.cc`, `PrimaryLogPG.cc`, and `semantic_op.cc` with the actual
   fork toolchain, buffer encoders, librados AIO API and generated configuration.
5. Run real result encode/decode roundtrips, old result/entry and truncated-wire
   rejection (including empty old result), actual OSD oversize/status faults,
   and RGW single-failure/partial/all-failed requests. Confirm HTTP500 rather
   than HTTP200 and completion/resource cleanup in the built chain.
6. Resolve already-audited handler registration, missing centroid population,
   per-object cluster identity and queue/request/data association before any
   golden-chain claim. They remain outside this bounded ABI follow-up.

Review remains bounded and parent verification is required. No git, commits,
downloads, network research, remote operations or hardware were used. Local
read-only explore agents were requested for failure-path and loader/shell review;
no substantive review deliverables were collected before this bounded handoff,
and remaining background work was cancelled. No agent review approval or
unavailable Oracle approval is claimed.
