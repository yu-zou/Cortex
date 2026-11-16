# Task15 prerequisite: integrated loader leases

2026-11-01 — **compiled isolated prerequisite PASS**, not PG/OSD async integration approval.

## Contract implemented

`CortexDriverLoader::Lease` is a typed, copyable/movable generation token created only by the loader. Its shared state contains the resolved immutable API, DSO handle, closed-admission bit and separate lock-free active-lease count. `SemanticCpuBackend` now takes this token and derives its function pointer from the token's table. It cannot accept an arbitrary `shared_ptr<void>` or unrelated function pointer as proof of DSO ownership.

* `try_acquire_lease(out)` is **startup-only**, outside PG/OSD/shard locks. It takes the lifecycle mutex; it is not a PG admission primitive. Nonempty output returns `-EINVAL`; absent/stopped generation returns `-ESHUTDOWN`.
* `request_unload()` is a lifecycle-coordinator operation outside those locks. It closes new admission, returns `-EBUSY` immediately after the short lifecycle critical section if leases exist, and never waits for a user to release a token. A stopped generation remains loaded and blocks replacement load with `-EBUSY`.
* Existing tokens, including copies made by existing owners after stop, retain their immutable function table until release. Lease destruction is `noexcept`: atomic release decrement and shared-state reference release only. No driver calls, loader callbacks, queue mutexes, posts, joins or waits. The loader retains state through retirement; state destruction itself is inert even if its last shared reference happens to disappear elsewhere.
* Only a later coordinator `request_unload()` with acquire-observed zero leases performs shutdown followed by `dlclose`. Last lease destruction does **not** unload. Token count is independent of executor task/byte accounting.
* Existing `void unload()` remains compatible when no leases exist. Busy unload logs refusal and preserves `is_loaded()` and raw API rather than silently clearing them. Static singleton teardown with live leases aborts instead of unloading active code or silently leaking a busy generation.
* Public raw `api` callers still require external call/unload coordination. `is_loaded()` is an atomic status observation, not a raw-call lifetime guard. Lifecycle entry points are startup/coordinator-only; driver init/shutdown under their mutex must not reenter loader lifecycle methods. No new PG→loader/driver lock edge is enabled.
* Version 2 gates, all-symbol-before-init resolution, failed-init cleanup and transactional API publication remain intact. State allocation is before `dlopen/init` and reports `-ENOMEM`.

## Exact changed source files

Relative to `/mnt/d/RAG/GitHubPrj/Cortex`:

1. `sys/src/cortex/loader/cortex_driver_loader.h`
2. `sys/src/cortex/loader/cortex_driver_loader.cc`
3. `sys/src/cortex/osd/semantic_cpu_backend.h`
4. `sys/src/cortex/osd/semantic_cpu_backend.cc`
5. `sys/src/cortex/osd/test_semantic_cpu_backend.cc`
6. `driver/test/test_loader_leases.cc` (new)
7. `driver/test/loader_lease_probe.h` (new)
8. `driver/test/loader_lease_stub.cc` (new)
9. `driver/test/CMakeLists.txt`

Harness `/tmp/opencode/cortex-task8-audit/CMakeLists.txt` additionally compiles the actual loader into the adapter test, links its existing real headers and creates the probe DSO. It remains an isolated harness, not a full Ceph build registration.

## Red → green evidence

Logs/XML: `/home/user/OpenCodePrj/.omo/evidence/cortex-plan/task-15/loader-*`.

1. **Original-source compile red:** `loader-leases-red.log` records actual compiler failures for missing `Lease`, `try_acquire_lease`, `request_unload`. This was run before loader implementation.
2. **Behavioral mutation red:** after implementation, temporarily bypassed only the active-lease guard in `request_unload()`, reproducing the old unsafe shutdown behavior. `loader-leases-race-red.log` records the barrier-controlled running backend: unload wrongly returns 0, replacement load wrongly succeeds, shutdown/unload counters become 1 while call is paused, then subprocess SIGSEGV on return into unloaded code. This is an explicit mutation control, **not** a claim that a lease test compiled on the original loader. Guard restored immediately; final normal and sanitizer builds rerun green.
3. **Final normal green:** GCC 9.4, C++17, default RTTI/exceptions; actual production loader/backend/executor sources, actual ABI headers and DSOs. `loader-leases-ctest.log`: **18/18 CTests**, zero failures. This is the existing 17-test harness plus the new focused loader target, not 18 new tests.
4. Focused GTest XML: `loader-leases-green.xml` **2/2**, `loader-backend-green.xml` **8/8**. Existing loader ABI target preserves version-1, missing-symbol, failed-init, busy-reload, eight-argument/full-result and actual uninitialized-driver checks.
5. **Sanitized green:** separate `lease-asan` build, `-fsanitize=address,undefined -fno-omit-frame-pointer`, executable `-no-pie`, `ASAN_OPTIONS=detect_leaks=1`: **4/4 filtered CTests** (loader ABI, loader leases, backend, executor), no diagnostics. Production loader/backend/executor, probe stub and driver/mock DSOs compiled with sanitizers; imported existing GTest/OpenSSL libraries are not instrumented. No full-suite/full-Ceph/TSan coverage claim. Initial sanitizer configure lacked OpenSSL discovery; corrected to the same explicit existing Conda OpenSSL root used by the normal harness.

The backend running-call test uses a condition-variable entry/release barrier, not sleep/race timing. It verifies input/accounting retained through cancellation/join, admission rejection after stop, unload/reload busy while paused, busy after join while adapter owns token, zero shutdown/deallocation/unload even after final adapter release, then explicit retry: shutdown counter/order → actual `delete` counter/order → real DSO destructor counter/order. Probe state lives in the executable after `dlclose`; the temporary `RTLD_NOLOAD` inspection handle is closed before the call. The second loader test transfers final token release to another thread and holds that thread alive while coordinator successfully unloads, plus copy/move/self-assignment coverage. Fresh load after retirement reopens admission.

## Reproduce

```sh
cmake -S /tmp/opencode/cortex-task8-audit -B /tmp/opencode/cortex-task8-audit/build -DOPENSSL_ROOT_DIR=/home/user/miniconda3
cmake --build /tmp/opencode/cortex-task8-audit/build -j2
ctest --test-dir /tmp/opencode/cortex-task8-audit/build --output-on-failure
cmake -S /tmp/opencode/cortex-task8-audit -B /tmp/opencode/cortex-task8-audit/lease-asan -DOPENSSL_ROOT_DIR=/home/user/miniconda3 -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined -no-pie" -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build /tmp/opencode/cortex-task8-audit/lease-asan --target test_loader_leases test_loader_abi test_semantic_cpu_backend test_semantic_executor -j2
ASAN_OPTIONS=detect_leaks=1 ctest --test-dir /tmp/opencode/cortex-task8-audit/lease-asan -R 'test_loader_|test_semantic_cpu_backend|test_semantic_executor' --output-on-failure
```

## Remaining integration / review boundary

No PG/OSD runtime paths enabled. Still required: real configured-Ceph compilation/relink of this delta, OSD startup lease acquisition and explicit adapter destruction before unload retry, PG OpFinisher/identity/generation hooks, callback final-retirement integration, normal and both fast shutdown coordinators and real-PG progress/cancel tests. A running backend that fails to return cannot be safely unloaded; these tests release their controlled barrier and make no bounded-hang-shutdown claim. The earlier first-ten FiQA adapter evidence predates this constructor change; this slice reruns actual compact-v2 adapter/executor correctness, not the full remote FiQA suite.

Review-work skill loaded. Its required Oracle/QA/context agent types and native spawn tools are unavailable in this tool set; all five independent review lanes are **INCONCLUSIVE / not approved**. Parent source/contract inspection and executable checks above passed; they do not substitute for the unavailable five-agent review. No plan checkboxes changed.
