"""Safety helpers: fail closed before touching anything outside the project.

The protected paths below are pre-existing assets that this project must never
write to (legacy datasets, the ps_fix project, CEMU VMs, Ceph state).
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROJECT_ROOT = Path("/home/chenming/strata_sim")

READ_ONLY_PREFIXES = (
    Path("/home/chenming/data"),
    Path("/home/chenming/vdb_baseline"),
    Path("/home/chenming/acc_top_vivado_test"),
    Path("/home/chenming/shared"),
    Path("/shared"),
    Path("/var/lib/ceph"),
    Path("/etc/ceph"),
    Path("/opt/pkg"),
    Path("/home/chenming/experiments"),
    Path("/home/chenming/ceph_rgw"),
)


class UnsafePathError(ValueError):
    """Raised when a path violates the project write/ownership contract."""


def assert_within_project(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve *path* and assert it is inside *project_root*.

    Returns the resolved path on success, raises :class:`UnsafePathError`
    otherwise. Never creates anything.
    """
    root = Path(project_root) if project_root is not None else DEFAULT_PROJECT_ROOT
    root = root.resolve()
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise UnsafePathError(f"{resolved} is outside project root {root}")
    for prefix in READ_ONLY_PREFIXES:
        p = prefix.resolve()
        if resolved == p or p in resolved.parents:
            raise UnsafePathError(f"{resolved} is under read-only protected path {p}")
    return resolved


def assert_writable_output(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Like assert_within_project but refuses paths that already exist as files."""
    resolved = assert_within_project(path, project_root)
    if resolved.exists() and resolved.is_file():
        raise UnsafePathError(f"refusing to overwrite existing file {resolved}")
    return resolved
