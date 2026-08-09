from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "project" / "data" / "manifests" / "reproducibility_inputs_v1.json"
SOURCE_MANIFEST = ROOT / "project" / "data" / "manifests" / "source_manifest_v1.json"
VALIDATOR = ROOT / "project" / "modeling" / "reproducibility.py"


def _load_validator():
    assert VALIDATOR.is_file(), "the fail-closed reproducibility validator is missing"
    spec = importlib.util.spec_from_file_location("practicum_reproducibility", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_covers_every_upstream_source_and_bounds_h4_claim() -> None:
    assert MANIFEST.is_file(), "the machine-checkable input receipt manifest is missing"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))

    source_ids = {item["id"] for item in source_manifest["sources"]}
    boundary_ids = {item["source_id"] for item in manifest["upstream_boundaries"]}
    assert boundary_ids == source_ids
    input_classes = {
        input_class
        for boundary in manifest["upstream_boundaries"]
        for input_class in boundary["input_classes"]
    }
    assert {"public", "restricted", "ignored", "missing", "mutable"} <= input_classes

    reviewed = manifest["scopes"]["reviewed-modeling"]
    assert reviewed["claim"] == "reviewed-output consistency"
    assert reviewed["full_upstream_reproduction_established"] is False
    assert "H4" in reviewed["description"]

    for receipt in manifest["receipts"]:
        assert receipt["path"].startswith("project/")
        assert receipt["version"]
        assert receipt["license"]
        assert receipt["git_state"] == "tracked"
        assert receipt["sha256_mode"] in {"bytes", "git-text-lf"}
        assert len(receipt["sha256"]) == 64


def test_reviewed_modeling_receipts_validate_from_the_current_checkout() -> None:
    validator = _load_validator()

    report = validator.validate_scope(
        root=ROOT,
        manifest_path=MANIFEST,
        scope="reviewed-modeling",
    )

    assert report["status"] == "ready", report
    assert report["claim"] == "reviewed-output consistency"
    assert report["full_upstream_reproduction_established"] is False
    assert len(report["checks"]) == 16
    assert {item["status"] for item in report["checks"]} == {"verified"}


def test_receipt_hash_drift_fails_closed(tmp_path: Path) -> None:
    validator = _load_validator()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["receipts"][0]["sha256"] = "0" * 64
    changed_manifest = tmp_path / "reproducibility_inputs_v1.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_scope(
        root=ROOT,
        manifest_path=changed_manifest,
        scope="reviewed-modeling",
    )

    assert report["status"] == "blocked"
    assert "checksum-mismatch" in {item["code"] for item in report["blockers"]}


def test_untracked_receipt_cannot_satisfy_a_clean_checkout_scope(tmp_path: Path) -> None:
    validator = _load_validator()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    input_path = tmp_path / "project" / "input.txt"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("local cache\n", encoding="utf-8")
    digest = validator.hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "scopes": {
            "reviewed-modeling": {
                "claim": "reviewed-output consistency",
                "full_upstream_reproduction_established": False,
                "receipt_ids": ["local-cache"],
                "limitations": [],
            }
        },
        "receipts": [
            {
                "id": "local-cache",
                "path": "project/input.txt",
                "version": "test-v1",
                "license": "test-only",
                "git_state": "tracked",
                "sha256": digest,
                "sha256_mode": "bytes",
                "canonical_bytes": len(input_path.read_bytes()),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_scope(
        root=tmp_path,
        manifest_path=manifest_path,
        scope="reviewed-modeling",
    )

    assert report["status"] == "blocked"
    assert report["full_upstream_reproduction_established"] is False
    assert "receipt-not-git-tracked" in {
        item["code"] for item in report["blockers"]
    }


def test_full_upstream_scope_reports_restricted_and_missing_receipt_blockers() -> None:
    validator = _load_validator()

    report = validator.validate_scope(
        root=ROOT,
        manifest_path=MANIFEST,
        scope="full-upstream",
    )

    assert report["status"] == "blocked"
    assert report["full_upstream_reproduction_established"] is False
    blocker_codes = {item["code"] for item in report["blockers"]}
    assert "restricted-input-git-tracked" in blocker_codes
    assert "missing-versioned-granule-receipt" in blocker_codes
    assert "mutable-source-snapshot-incomplete" in blocker_codes
    assert "catalog-selection-required" in blocker_codes
    restricted_tracking = [
        item
        for item in report["blockers"]
        if item["code"] == "restricted-input-git-tracked"
    ]
    assert restricted_tracking == [
        {
            "code": "restricted-input-git-tracked",
            "detail": (
                "A partner-restricted input is tracked by Git and violates the "
                "declared public/restricted boundary."
            ),
            "source_id": "eagle_i",
            "path": "project/data/raw/Outage_Dataset_R1",
        }
    ]
    assert "ignored-storage-policy-mismatch" not in blocker_codes
