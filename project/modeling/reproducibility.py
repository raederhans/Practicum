#!/usr/bin/env python3
"""Validate modeling input receipts without importing the modeling stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "project" / "data" / "manifests" / "reproducibility_inputs_v1.json"
ALLOWED_DISPOSITIONS = {
    "resolved",
    "externally-actionable",
    "permission-gated",
    "unavailable",
    "ambiguous",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(path: Path, mode: str) -> bytes:
    content = path.read_bytes()
    if mode == "bytes":
        return content
    if mode == "git-text-lf":
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    raise ValueError(f"Unsupported sha256_mode: {mode}")


def _git_tracks(root: Path, relative_path: str) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", relative_path],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _git_ignores(root: Path, relative_path: str) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", "--no-index", relative_path],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    return None


def _git_tracked_paths(root: Path, relative_path: str) -> list[str] | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", relative_path],
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return sorted(
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    )


def _tracked_tree_inventory(
    root: Path, relative_path: str, algorithm: str
) -> dict[str, Any]:
    if algorithm != "sha256-lf-lines-v1":
        raise ValueError(f"Unsupported tracked-tree algorithm: {algorithm}")
    paths = _git_tracked_paths(root, relative_path)
    if paths is None:
        raise RuntimeError("Git tracking state could not be verified.")

    lines: list[str] = []
    total_bytes = 0
    for path in paths:
        target = root / path
        if not target.is_file():
            raise FileNotFoundError(path)
        content = _canonical_bytes(target, "git-text-lf")
        total_bytes += len(content)
        digest = hashlib.sha256(content).hexdigest()
        lines.append(f"{path}\t{len(content)}\t{digest}\n")

    inventory = "".join(lines).encode("utf-8")
    return {
        "path_count": len(paths),
        "total_bytes": total_bytes,
        "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
    }


def _blocker(code: str, detail: str, **context: Any) -> dict[str, Any]:
    return {"code": code, "detail": detail, **context}


def validate_scope(
    *,
    root: Path = ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
    scope: str = "reviewed-modeling",
) -> dict[str, Any]:
    """Return a deterministic report; any missing or drifting requirement blocks."""

    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json(manifest_path)
    scopes = manifest.get("scopes", {})
    if scope not in scopes:
        raise ValueError(f"Unknown reproducibility scope: {scope}")

    scope_config = scopes[scope]
    blockers: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []

    receipt_items = manifest.get("receipts", [])
    receipts: dict[str, dict[str, Any]] = {}
    for item in receipt_items:
        receipt_id = item.get("id")
        if not receipt_id or receipt_id in receipts:
            blockers.append(
                _blocker(
                    "invalid-receipt-id",
                    "Every receipt must have a unique non-empty id.",
                    receipt_id=receipt_id,
                )
            )
            continue
        receipts[receipt_id] = item

    for receipt_id in scope_config.get("receipt_ids", []):
        receipt = receipts.get(receipt_id)
        if receipt is None:
            blockers.append(
                _blocker(
                    "receipt-not-declared",
                    "The selected scope references an undeclared receipt.",
                    receipt_id=receipt_id,
                )
            )
            continue

        relative_path = receipt.get("path", "")
        target = root / relative_path
        if not target.is_file():
            blockers.append(
                _blocker(
                    "input-missing",
                    "A required receipted input is missing.",
                    receipt_id=receipt_id,
                    path=relative_path,
                )
            )
            continue

        try:
            content = _canonical_bytes(target, receipt.get("sha256_mode", ""))
        except ValueError as exc:
            blockers.append(
                _blocker(
                    "unsupported-hash-mode",
                    str(exc),
                    receipt_id=receipt_id,
                    path=relative_path,
                )
            )
            continue

        actual_hash = hashlib.sha256(content).hexdigest()
        actual_bytes = len(content)
        if actual_hash != receipt.get("sha256"):
            blockers.append(
                _blocker(
                    "checksum-mismatch",
                    "A required input does not match its SHA-256 receipt.",
                    receipt_id=receipt_id,
                    path=relative_path,
                    expected_sha256=receipt.get("sha256"),
                    actual_sha256=actual_hash,
                )
            )
            continue
        if actual_bytes != receipt.get("canonical_bytes"):
            blockers.append(
                _blocker(
                    "size-mismatch",
                    "A required input does not match its canonical byte count.",
                    receipt_id=receipt_id,
                    path=relative_path,
                    expected_bytes=receipt.get("canonical_bytes"),
                    actual_bytes=actual_bytes,
                )
            )
            continue

        expected_git_state = receipt.get("git_state")
        if expected_git_state != "tracked":
            blockers.append(
                _blocker(
                    "invalid-receipt-git-state",
                    "A clean-checkout receipt must explicitly require a tracked file.",
                    receipt_id=receipt_id,
                    path=relative_path,
                )
            )
            continue
        tracked = _git_tracks(root, relative_path)
        if tracked is None:
            blockers.append(
                _blocker(
                    "git-state-unverifiable",
                    "Git tracking state could not be verified for a receipted input.",
                    receipt_id=receipt_id,
                    path=relative_path,
                )
            )
            continue
        if not tracked:
            blockers.append(
                _blocker(
                    "receipt-not-git-tracked",
                    "A required input exists locally but would be absent from a clean checkout.",
                    receipt_id=receipt_id,
                    path=relative_path,
                )
            )
            continue

        checks.append(
            {
                "receipt_id": receipt_id,
                "path": relative_path,
                "status": "verified",
                "sha256": actual_hash,
                "canonical_bytes": actual_bytes,
            }
        )

    if scope == "full-upstream":
        source_manifest_path = root / manifest.get("source_manifest_path", "")
        if not source_manifest_path.is_file():
            blockers.append(
                _blocker(
                    "source-manifest-missing",
                    "The upstream source catalog is missing.",
                    path=str(source_manifest_path),
                )
            )
            source_by_id: dict[str, dict[str, Any]] = {}
        else:
            source_manifest = _load_json(source_manifest_path)
            source_by_id = {
                item.get("id"): item
                for item in source_manifest.get("sources", [])
                if item.get("id")
            }

        tree_receipts: dict[str, dict[str, Any]] = {}
        verified_tree_receipts: set[str] = set()
        for tree_receipt in manifest.get("tracked_tree_receipts", []):
            receipt_id = tree_receipt.get("id")
            if not receipt_id or receipt_id in tree_receipts:
                blockers.append(
                    _blocker(
                        "invalid-tracked-tree-receipt-id",
                        "Every tracked-tree receipt must have a unique non-empty id.",
                        receipt_id=receipt_id,
                    )
                )
                continue
            tree_receipts[receipt_id] = tree_receipt
            if tree_receipt.get("git_state") != "tracked":
                blockers.append(
                    _blocker(
                        "invalid-tracked-tree-git-state",
                        "A tracked-tree receipt must explicitly require tracked files.",
                        receipt_id=receipt_id,
                    )
                )
                continue
            try:
                inventory = _tracked_tree_inventory(
                    root,
                    tree_receipt.get("path", ""),
                    tree_receipt.get("inventory_algorithm", ""),
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                blockers.append(
                    _blocker(
                        "tracked-tree-unverifiable",
                        str(exc),
                        receipt_id=receipt_id,
                        path=tree_receipt.get("path"),
                    )
                )
                continue

            for field, code in (
                ("path_count", "tracked-tree-count-mismatch"),
                ("total_bytes", "tracked-tree-size-mismatch"),
                ("inventory_sha256", "tracked-tree-checksum-mismatch"),
            ):
                if inventory[field] != tree_receipt.get(field):
                    blockers.append(
                        _blocker(
                            code,
                            "A tracked-tree input does not match its aggregate receipt.",
                            receipt_id=receipt_id,
                            path=tree_receipt.get("path"),
                            expected=tree_receipt.get(field),
                            actual=inventory[field],
                        )
                    )
                    break
            else:
                verified_tree_receipts.add(receipt_id)
                checks.append(
                    {
                        "receipt_id": receipt_id,
                        "path": tree_receipt.get("path"),
                        "status": "verified",
                        "sha256": inventory["inventory_sha256"],
                        "canonical_bytes": inventory["total_bytes"],
                        "path_count": inventory["path_count"],
                        "receipt_kind": "tracked-tree",
                    }
                )

        boundaries = manifest.get("upstream_boundaries", [])
        boundary_source_ids = [item.get("source_id") for item in boundaries]
        boundary_ids = set(boundary_source_ids)
        if (
            boundary_ids != set(source_by_id)
            or len(boundary_source_ids) != len(boundary_ids)
        ):
            blockers.append(
                _blocker(
                    "source-boundary-coverage-mismatch",
                    "Every upstream source must have exactly one audited boundary entry.",
                )
            )

        for boundary in boundaries:
            source_id = boundary.get("source_id")
            source = source_by_id.get(source_id, {})
            disposition = boundary.get("disposition")
            full_rerun_ready = boundary.get("full_rerun_ready", False)
            missing_boundary_metadata = [
                field
                for field in ("detail", "next_action", "evidence")
                if not boundary.get(field)
            ]
            if not full_rerun_ready and not boundary.get("blocker_code"):
                missing_boundary_metadata.append("blocker_code")
            if missing_boundary_metadata:
                blockers.append(
                    _blocker(
                        "source-boundary-metadata-incomplete",
                        "An upstream boundary lacks required blocker, action, or evidence metadata.",
                        source_id=source_id,
                        missing_fields=missing_boundary_metadata,
                    )
                )
            dispositions.append(
                {
                    "source_id": source_id,
                    "disposition": disposition,
                    "full_rerun_ready": full_rerun_ready,
                    "blocker_code": boundary.get("blocker_code"),
                }
            )
            if disposition not in ALLOWED_DISPOSITIONS:
                blockers.append(
                    _blocker(
                        "invalid-boundary-disposition",
                        "Every upstream boundary must use an allowed disposition.",
                        source_id=source_id,
                        disposition=disposition,
                    )
                )
            elif full_rerun_ready != (disposition == "resolved"):
                blockers.append(
                    _blocker(
                        "boundary-disposition-readiness-mismatch",
                        "Resolved disposition and full-rerun readiness must agree.",
                        source_id=source_id,
                        disposition=disposition,
                        full_rerun_ready=full_rerun_ready,
                    )
                )
            missing_metadata = [
                field
                for field in ("status", "version", "license", "reproducibility")
                if not source.get(field)
            ]
            if missing_metadata:
                blockers.append(
                    _blocker(
                        "source-metadata-incomplete",
                        "An upstream source lacks required version/license/status metadata.",
                        source_id=source_id,
                        missing_fields=missing_metadata,
                    )
                )

            for receipt_id in boundary.get("satisfied_by_receipt_ids", []):
                if not any(check["receipt_id"] == receipt_id for check in checks):
                    blockers.append(
                        _blocker(
                            "boundary-receipt-unverified",
                            "A ready upstream boundary depends on an unverified receipt.",
                            source_id=source_id,
                            receipt_id=receipt_id,
                        )
                    )

            tree_receipt_id = boundary.get("tracked_tree_receipt_id")
            if tree_receipt_id and tree_receipt_id not in verified_tree_receipts:
                blockers.append(
                    _blocker(
                        "boundary-receipt-unverified",
                        "An upstream boundary depends on an unverified tracked-tree receipt.",
                        source_id=source_id,
                        receipt_id=tree_receipt_id,
                    )
                )

            if not full_rerun_ready:
                blockers.append(
                    _blocker(
                        boundary.get("blocker_code", "upstream-boundary-not-ready"),
                        boundary.get("detail", "The upstream boundary is not ready."),
                        source_id=source_id,
                        disposition=disposition,
                    )
                )

            if boundary.get("required_git_state") == "untracked":
                tracked = _git_tracks(root, boundary.get("git_path", ""))
                if tracked is None:
                    blockers.append(
                        _blocker(
                            "git-state-unverifiable",
                            "Git tracking state could not be verified for a restricted input.",
                            source_id=source_id,
                            path=boundary.get("git_path"),
                        )
                    )
                elif tracked:
                    blockers.append(
                        _blocker(
                            "restricted-input-git-tracked",
                            "A partner-restricted input is tracked by Git and violates the declared public/restricted boundary.",
                            source_id=source_id,
                            path=boundary.get("git_path"),
                        )
                    )
            elif boundary.get("required_git_state") == "ignored":
                ignored = _git_ignores(root, boundary.get("git_path", ""))
                if ignored is None:
                    blockers.append(
                        _blocker(
                            "git-state-unverifiable",
                            "Git ignore state could not be verified for a local cache boundary.",
                            source_id=source_id,
                            path=boundary.get("git_path"),
                        )
                    )
                elif not ignored:
                    blockers.append(
                        _blocker(
                            "ignored-storage-policy-mismatch",
                            "A boundary declared as ignored local cache is not covered by Git ignore rules.",
                            source_id=source_id,
                            path=boundary.get("git_path"),
                        )
                    )

    return {
        "schema_version": manifest.get("schema_version"),
        "scope": scope,
        "status": "blocked" if blockers else "ready",
        "claim": scope_config.get("claim"),
        "full_upstream_reproduction_established": scope_config.get(
            "full_upstream_reproduction_established"
        ),
        "checks": checks,
        "blockers": blockers,
        "dispositions": dispositions,
        "limitations": scope_config.get("limitations", []),
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"scope={report['scope']}")
    print(f"status={report['status']}")
    print(f"claim={report['claim']}")
    print(f"verified_receipts={len(report['checks'])}")
    for item in report["blockers"]:
        context = " ".join(
            f"{key}={value}"
            for key, value in item.items()
            if key not in {"code", "detail"}
        )
        suffix = f" {context}" if context else ""
        print(f"BLOCKED code={item['code']}{suffix}: {item['detail']}")
    for item in report["dispositions"]:
        print(
            "DISPOSITION "
            f"source_id={item['source_id']} "
            f"value={item['disposition']} "
            f"full_rerun_ready={item['full_rerun_ready']}"
        )
    for limitation in report["limitations"]:
        print(f"LIMITATION: {limitation}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation of Practicum modeling input receipts."
    )
    parser.add_argument(
        "--scope",
        choices=("reviewed-modeling", "full-upstream"),
        default="reviewed-modeling",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_scope(root=ROOT, manifest_path=args.manifest, scope=args.scope)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
