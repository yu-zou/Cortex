"""Execute bounded, unchanged entrypoint fragments; never start a cluster."""
import os
import pathlib
import subprocess
import sys
import tempfile

source = pathlib.Path(sys.argv[1]).read_text()
driver, old_driver = sys.argv[2:4]
probe = source.split("ABI_OUTPUT=$(", 1)[1].split(") || true", 1)[0]
for library, success in [(driver, True), (old_driver, False)]:
    with tempfile.TemporaryDirectory(prefix="cortex-abi-probe-") as directory:
        pathlib.Path(directory, "libcortex_driver.so").symlink_to(library)
        env = dict(os.environ, LD_LIBRARY_PATH=directory)
        result = subprocess.run(["bash", "-c", probe], env=env,
                                capture_output=True, text=True)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        assert f"ABI_VERSION={2 if success else 1}" in result.stdout

# Execute the actual final gate and success marker/exit, not a rewritten policy.
tail = source[source.index('[ "$FPGA_FAIL" -eq 0 ]'):]
for failures in [0, 1, 3]:
    result = subprocess.run(["bash", "-c", tail],
                            env=dict(os.environ, FPGA_FAIL=str(failures)),
                            capture_output=True, text=True)
    assert (result.returncode == 0) == (failures == 0), result.stdout
    assert ("ALL TESTS PASSED" in result.stdout) == (failures == 0)
print("entrypoint ABI 2/1 probes and final FPGA_FAIL 0/1/3 gate passed")
