#!/usr/bin/env bash
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
name=daisyplus-lean-licensed
out=${BUILD_DIR:-"$root/lean-build-helper-off"}
out=$(realpath -m -- "$out")
tool=${VIVADO_TOOL_ROOT:-./vivado-tool}
runtime=${VIVADO_RUNTIME_ROOT:-./vivado-runtime}
license_dir=${VIVADO_LICENSE_DIR:-./licenses}
[[ $out == "$root/"* ]] || { printf 'BUILD_DIR must be within %s\n' "$root" >&2; exit 1; }
for path in "$tool/Vivado/2019.1/bin/vivado" "$runtime/x86_64-linux-gnu" "$runtime/lib" "$license_dir/Xilinx.lic" "$license_dir/vivado_license_all.lic"; do
    [[ -e $path ]] || { printf 'Missing dependency: %s\n' "$path" >&2; exit 1; }
done
exec 9>"$root/.container-build.lock"
flock -n 9 || { printf 'Another package launch is preparing\n' >&2; exit 1; }
if docker ps --format '{{.Names}}' | grep -Eq '^daisyplus-'; then
    printf 'An isolated DaisyPlus build is already active\n' >&2
    exit 1
fi
BUILD_DIR="$out" bash "$root/build.sh" prepare
sha256sum "$root/build.sh" "$root/build.tcl" "$root/synth_pre.tcl" "$root/gates.tcl" "$root/sources.list" "$root/check_sources.py" "$root/run_container.sh" > "$out/scripts.sha256"
container_out="/work/${out#"$root/"}"
docker run -d --name "$name" --network none --cpus 8 --memory 40g \
    --user "$(id -u):$(id -g)" --workdir "$container_out" \
    --mount "type=bind,src=$root,dst=/work" \
    --mount "type=bind,src=$tool,dst=/opt/pkg/Vivado2019.1,readonly" \
    --mount "type=bind,src=$runtime/x86_64-linux-gnu,dst=/usr/lib/x86_64-linux-gnu,readonly" \
    --mount "type=bind,src=$runtime/lib,dst=/lib/x86_64-linux-gnu,readonly" \
    --mount "type=bind,src=$license_dir/Xilinx.lic,dst=/licenses/Xilinx.lic,readonly" \
    --mount "type=bind,src=$license_dir/vivado_license_all.lic,dst=/licenses/vivado_license_all.lic,readonly" \
    -e XILINXD_LICENSE_FILE=/licenses/Xilinx.lic:/licenses/vivado_license_all.lic \
    -e "HOME=$container_out" -e "TMPDIR=$container_out" ubuntu:16.04 \
    timeout -k 30s 24h /opt/pkg/Vivado2019.1/Vivado/2019.1/bin/vivado \
    -mode batch -source /work/build.tcl -log "$container_out/vivado.log" \
    -journal "$container_out/vivado.jou" -tclargs "$container_out" | tee "$out/container.id"
printf 'Detached build: %s; logs: %s/vivado.log\n' "$name" "$out"
