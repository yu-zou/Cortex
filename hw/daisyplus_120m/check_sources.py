#!/usr/bin/env python3
"""Audit prepared source inputs without starting Vivado."""
import json
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1]).resolve() / "src/daisyplus_xdma2_bd_bundle_slim"
package = Path(__file__).resolve().parent
for relative in (package / "sources.list").read_text().splitlines():
    assert (root / relative).exists(), f"Missing source: {relative}"
files = [p for p in root.rglob("*") if p.is_file()]
for path in files:
    assert not path.is_symlink(), f"Unexpected symlink: {path}"
    assert path.suffix.lower() not in {".dcp", ".edf", ".edn", ".bit"}, path
    assert "_sim_netlist" not in path.name, path
ip = root / "shared/acc_top_repack_ip"
ns = {"s": "http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"}
component = ET.parse(ip / "component.xml")
references = component.findall(".//s:file/s:name", ns)
assert references, "No IP-XACT file references"
for node in references:
    path = (ip / node.text).resolve()
    assert path.is_relative_to(ip), f"External custom-IP reference: {node.text}"
    assert path.is_file(), f"Missing custom-IP input: {node.text}"
for path in ip.rglob("*.xci"):
    ET.parse(path)
bd_path = next(root.rglob("design_1.bd"))
bd = json.loads(bd_path.read_text())["design"]
components = bd["components"]
for name in ("xdma_0", "xdma_1", "xdma_2", "ddr4_0", "acc_top_0", "blk_mem_gen_0"):
    assert name in components, f"Missing BD cell: {name}"
assert components["acc_top_0"]["vlnv"] == "ivfpq:hls:acc_top:1.4"
assert float(components["clk_wiz_0"]["parameters"]["CLKOUT1_REQUESTED_OUT_FREQ"]["value"]) == 120
assert not any("/shared/" in p.read_text(errors="replace") for p in [bd_path, ip / "component.xml"]), "Absolute workstation reference"
print(f"SOURCE_STATIC_CHECK_PASS: {len(files)} files, {sum(p.stat().st_size for p in files)} bytes; {len(references)} IP-XACT references")
with (Path(sys.argv[1]).resolve() / "source.sha256").open("w") as output:
    for path in sorted(files):
        output.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n")
