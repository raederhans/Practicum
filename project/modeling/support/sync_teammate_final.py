#!/usr/bin/env python3
"""Safely inspect or apply a commit-pinned teammate-final synchronization plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = ROOT / "project" / "data" / "manifests"
DEFAULT_PLAN = MANIFEST_DIR / "sync_plan_v1.json"
DEFAULT_INVENTORY = MANIFEST_DIR / "sync_inventory_v1.json"
EXPECTED_DONOR_COMMIT = "1f63e190ce280852d68945dbfce486075adda69b"


class SyncError(RuntimeError):
    """The synchronization contract or Git evidence is invalid."""


class SyncConflictError(SyncError):
    """At least one target has different local content; nothing was copied."""


def _git(git_root: Path, *args: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args],
        cwd=git_root,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise SyncError(f"git {' '.join(args)} failed: {message or 'unknown Git error'}")
    return result.stdout.decode("utf-8", errors="strict") if text else result.stdout


def validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise SyncError("planned paths must be non-empty strings")
    if "\\" in value or "//" in value:
        raise SyncError(f"path is not canonical POSIX syntax: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or (path.parts and path.parts[0].endswith(":")) or str(path) != value:
        raise SyncError(f"path is not a canonical relative POSIX path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SyncError(f"path contains an unsafe segment: {value!r}")
    return value


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != 1:
        raise SyncError("sync plan schema_version must be 1")
    if plan.get("donor_commit") != EXPECTED_DONOR_COMMIT:
        raise SyncError(f"donor_commit must remain pinned to {EXPECTED_DONOR_COMMIT}")
    ref = plan.get("reachability_ref")
    if not isinstance(ref, str) or not ref.strip():
        raise SyncError("reachability_ref is required")
    slices = plan.get("slices")
    if not isinstance(slices, dict):
        raise SyncError("slices must be an object")
    roots = slices.get("roots", [])
    files = slices.get("files", [])
    protected = plan.get("protected_prefixes", [])
    if not all(isinstance(items, list) for items in (roots, files, protected)):
        raise SyncError("roots, files, and protected_prefixes must be arrays")
    for value in [*roots, *files, *protected]:
        validate_relative_path(value)
    for label, values in (("roots", roots), ("files", files), ("protected_prefixes", protected)):
        if len(values) != len(set(values)):
            raise SyncError(f"duplicate path in {label}")
    overlap = set(roots) & set(files)
    if overlap:
        raise SyncError(f"paths cannot be both roots and files: {sorted(overlap)}")
    for planned in [*roots, *files]:
        for prefix in protected:
            if _is_under(planned, prefix) or _is_under(prefix, planned):
                raise SyncError(f"planned path intersects protected prefix: {planned} / {prefix}")
    return plan


def _verify_commit_and_ref(git_root: Path, commit: str, ref: str) -> None:
    _git(git_root, "cat-file", "-e", f"{commit}^{{commit}}")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, ref],
        cwd=git_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SyncError(f"pinned donor commit {commit} is not reachable from {ref}")


def _expand_plan(git_root: Path, plan: dict[str, Any]) -> list[tuple[str, str]]:
    commit = plan["donor_commit"]
    protected = plan["protected_prefixes"]
    expanded: list[tuple[str, str]] = []
    seen: set[str] = set()
    for root in plan["slices"]["roots"]:
        raw = _git(git_root, "ls-tree", "-r", "-z", "--name-only", commit, "--", root)
        assert isinstance(raw, bytes)
        paths = [item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item]
        if not paths:
            raise SyncError(f"planned root is absent at donor commit: {root}")
        for path in paths:
            validate_relative_path(path)
            if not _is_under(path, root):
                raise SyncError(f"Git returned path outside planned root {root}: {path}")
            if path in seen:
                raise SyncError(f"duplicate expanded donor path: {path}")
            seen.add(path)
            expanded.append((path, root))
    for path in plan["slices"]["files"]:
        if path in seen:
            raise SyncError(f"duplicate expanded donor path: {path}")
        _git(git_root, "cat-file", "-e", f"{commit}:{path}")
        seen.add(path)
        expanded.append((path, "explicit-files"))
    for path, _slice in expanded:
        if any(_is_under(path, prefix) for prefix in protected):
            raise SyncError(f"expanded donor path is protected: {path}")
    return sorted(expanded)


def _safe_target(target_root: Path, relative: str) -> Path:
    root = target_root.resolve()
    candidate = target_root.joinpath(*PurePosixPath(relative).parts)
    for parent in [candidate, *candidate.parents]:
        if parent == target_root.parent:
            break
        if parent.exists() and parent.is_symlink():
            raise SyncError(f"target path traverses a symlink: {relative}")
        if parent == target_root:
            break
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SyncError(f"target escapes target root: {relative}") from exc
    return candidate


def _blob(git_root: Path, commit: str, path: str) -> tuple[str, bytes]:
    oid = _git(git_root, "rev-parse", f"{commit}:{path}", text=True)
    assert isinstance(oid, str)
    oid = oid.strip()
    kind = _git(git_root, "cat-file", "-t", oid, text=True)
    assert isinstance(kind, str)
    if kind.strip() != "blob":
        raise SyncError(f"planned donor object is not a file blob: {path}")
    content = _git(git_root, "cat-file", "blob", oid)
    assert isinstance(content, bytes)
    return oid, content


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def synchronize(
    *,
    git_root: Path,
    target_root: Path,
    plan: dict[str, Any],
    apply: bool,
    inventory_path: Path,
) -> dict[str, Any]:
    git_root = Path(git_root)
    target_root = Path(target_root)
    inventory_path = Path(inventory_path)
    validate_plan(plan)
    commit = plan["donor_commit"]
    _verify_commit_and_ref(git_root, commit, plan["reachability_ref"])
    expanded = _expand_plan(git_root, plan)

    prepared: list[tuple[dict[str, Any], bytes, Path]] = []
    conflicts: list[str] = []
    for relative, slice_name in expanded:
        oid, content = _blob(git_root, commit, relative)
        target = _safe_target(target_root, relative)
        if target.exists():
            if not target.is_file() or target.is_symlink():
                status = "conflict"
            else:
                status = "skip-identical" if target.read_bytes() == content else "conflict"
        else:
            status = "planned-copy" if apply else "check-only"
        action = "skip-identical" if status == "skip-identical" else ("conflict" if status == "conflict" else "copy")
        entry = {
            "path": relative,
            "slice": slice_name,
            "donor_commit": commit,
            "git_oid": oid,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "action": action,
            "status": status,
        }
        prepared.append((entry, content, target))
        if status == "conflict":
            conflicts.append(relative)

    inventory: dict[str, Any] = {
        "schema_version": 1,
        "mode": "apply" if apply else "check",
        "donor_commit": commit,
        "reachability_ref": plan["reachability_ref"],
        "files": [entry for entry, _content, _target in prepared],
        "summary": {
            "total": len(prepared),
            "copy": sum(entry["action"] == "copy" for entry, _, _ in prepared),
            "skip_identical": sum(entry["action"] == "skip-identical" for entry, _, _ in prepared),
            "conflicts": len(conflicts),
        },
    }
    if conflicts:
        _write_json_atomic(inventory_path, inventory)
        raise SyncConflictError("conflicting local targets; no donor files copied: " + ", ".join(conflicts))
    if not apply:
        _write_json_atomic(inventory_path, inventory)
        return inventory

    target_root.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="teammate-final-stage-", dir=target_root) as temp_dir:
        stage_root = Path(temp_dir)
        for entry, content, _target in prepared:
            staged = stage_root.joinpath(*PurePosixPath(entry["path"]).parts)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(content)
            if hashlib.sha256(staged.read_bytes()).hexdigest() != entry["sha256"]:
                raise SyncError(f"staged blob verification failed: {entry['path']}")
        try:
            for entry, content, target in prepared:
                if entry["action"] == "skip-identical":
                    continue
                if target.exists():
                    if not target.is_file() or target.is_symlink() or target.read_bytes() != content:
                        raise SyncConflictError(f"target changed after preflight: {entry['path']}")
                    entry["action"] = "skip-identical"
                    entry["status"] = "skip-identical"
                    continue
                staged = stage_root.joinpath(*PurePosixPath(entry["path"]).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged, target)
                copied.append(target)
                entry["status"] = "copied"
        except BaseException:
            for path in reversed(copied):
                path.unlink(missing_ok=True)
            raise

    inventory["summary"] = {
        "total": len(prepared),
        "copied": sum(entry["status"] == "copied" for entry, _, _ in prepared),
        "skip_identical": sum(entry["status"] == "skip-identical" for entry, _, _ in prepared),
        "conflicts": 0,
    }
    _write_json_atomic(inventory_path, inventory)
    return inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="inspect only (default)")
    mode.add_argument("--apply", action="store_true", help="copy only after a complete clean preflight")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--git-root", type=Path, default=ROOT)
    parser.add_argument("--target-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        inventory = synchronize(
            git_root=args.git_root,
            target_root=args.target_root,
            plan=plan,
            apply=args.apply,
            inventory_path=args.inventory,
        )
    except (OSError, json.JSONDecodeError, SyncError) as exc:
        print(f"ERROR: {exc}")
        return 2
    summary = inventory["summary"]
    print(
        f"{inventory['mode'].upper()} OK: total={summary['total']} "
        f"copied={summary.get('copied', 0)} "
        f"copy_candidates={summary.get('copy', 0)} "
        f"skip_identical={summary['skip_identical']} conflicts={summary['conflicts']}"
    )
    print(f"Inventory: {args.inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
