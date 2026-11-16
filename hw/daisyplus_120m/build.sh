#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
archive=${SOURCE_ARCHIVE:-./daisyplus_xdma2_bd_bundle_slim.tar.zst}
vivado=${VIVADO:-vivado}
out=${BUILD_DIR:-"$root/build"}
mode=${1:-full}
case "$mode" in full|prepare) ;; *) printf 'Usage: bash build.sh [full|prepare]\n' >&2; exit 2;; esac
out=$(realpath -m -- "$out")
expected=c7a3342b32bb051e88496ac68d53d3e8d7f729dc73cc86ec4f6f6d2d7bc11d5f

if [[ $mode == full ]]; then
    command -v "$vivado" >/dev/null || { printf 'Vivado not found: %s\n' "$vivado" >&2; exit 1; }
fi
command -v tar >/dev/null || { printf 'tar is required\n' >&2; exit 1; }
command -v sha256sum >/dev/null || { printf 'sha256sum is required\n' >&2; exit 1; }
command -v python3 >/dev/null || { printf 'Python 3.9+ is required\n' >&2; exit 1; }
[[ -f $archive ]] || { printf 'Source archive not found: %s\n' "$archive" >&2; exit 1; }
actual=$(sha256sum "$archive")
actual=${actual%% *}
[[ $actual == "$expected" ]] || { printf 'Source archive SHA-256 mismatch: %s\n' "$actual" >&2; exit 1; }
[[ ! -e $out && ! -L $out ]] || { printf 'Refusing existing output directory: %s\n' "$out" >&2; exit 1; }
mkdir -p "$(dirname -- "$out")"
mkdir "$out"
mkdir "$out/src"

mapfile -t sources < "$root/sources.list"
for i in "${!sources[@]}"; do
    sources[$i]="daisyplus_xdma2_bd_bundle_slim/${sources[$i]}"
done
tar --zstd -xf "$archive" -C "$out/src" --exclude='*.dcp' --exclude='*_sim_netlist.*' --exclude='*.hwdef' "${sources[@]}"
python3 "$root/check_sources.py" "$out"
if [[ $mode == prepare ]]; then
    printf 'SOURCE_PREPARATION_COMPLETE: %s\n' "$out"
    exit 0
fi

vivado_version=$("$vivado" -version 2>&1)
if [[ $vivado_version != *"Vivado v2019.1"* ]]; then
    printf 'Vivado 2019.1 required; got:\n%s\n' "$vivado_version" >&2
    exit 1
fi

cd "$out"
"$vivado" -mode batch -source "$root/build.tcl" -log "$out/vivado.log" -journal "$out/vivado.jou" -tclargs "$out"
