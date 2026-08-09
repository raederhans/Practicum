from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "project" / "data" / "manifests" / "reproducibility_inputs_v1.json"
SOURCE_MANIFEST = ROOT / "project" / "data" / "manifests" / "source_manifest_v1.json"
VALIDATOR = ROOT / "project" / "modeling" / "reproducibility.py"
ALLOWED_DISPOSITIONS = {
    "resolved",
    "externally-actionable",
    "permission-gated",
    "unavailable",
    "ambiguous",
}


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
    assert {
        item["disposition"] for item in manifest["upstream_boundaries"]
    } <= ALLOWED_DISPOSITIONS
    assert all(item["next_action"] for item in manifest["upstream_boundaries"])
    assert all(item["evidence"] for item in manifest["upstream_boundaries"])
    assert all(
        item["disposition"] == "resolved"
        for item in manifest["upstream_boundaries"]
        if item["full_rerun_ready"]
    )
    assert all(
        item["disposition"] != "resolved"
        for item in manifest["upstream_boundaries"]
        if not item["full_rerun_ready"]
    )
    input_classes = {
        input_class
        for boundary in manifest["upstream_boundaries"]
        for input_class in boundary["input_classes"]
    }
    assert {
        "public",
        "tracked",
        "ambiguous",
        "permission-gated",
        "ignored",
        "missing",
        "mutable",
    } <= input_classes
    assert "restricted" not in input_classes

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


def test_eagle_i_public_release_and_tracked_derivatives_stay_distinct() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    sources = {item["id"]: item for item in source_manifest["sources"]}
    boundaries = {
        item["source_id"]: item for item in manifest["upstream_boundaries"]
    }

    eagle = sources["eagle_i"]
    release = eagle["official_release"]
    assert eagle["status"] == "public-release-local-derivatives-lineage-unproven"
    assert release["article_id"] == 24_237_376
    assert release["version"] == 4
    assert release["versioned_doi"] == "10.6084/m9.figshare.24237376.v4"
    assert release["license"] == "CC BY 4.0"
    assert release["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert len(release["files_used_by_repository_year_range"]) == 10
    assert {
        item["name"] for item in release["files_used_by_repository_year_range"]
    } == {f"eaglei_outages_{year}.csv" for year in range(2014, 2024)}
    assert all(
        re.fullmatch(r"[0-9a-f]{32}", item["md5"])
        for item in release["files_used_by_repository_year_range"]
    )

    boundary = boundaries["eagle_i"]
    assert boundary["input_classes"] == ["public", "tracked", "ambiguous"]
    assert boundary["disposition"] == "ambiguous"
    assert boundary["blocker_code"] == "tracked-derived-lineage-unproven"
    assert boundary["tracked_tree_receipt_id"] == "eagle-i-tracked-derivatives-v1"
    assert "restricted-input-git-tracked" in boundary["resolved_blocker_codes"]
    assert (
        "repository-commit:13f1a7bb60fe6c23c7aba9910f3d1fd04f050fcd"
        in boundary["evidence"]
    )
    assert not any(
        item.startswith("repository-commit:bf1a7e3")
        for item in boundary["evidence"]
    )

    tree_receipt = {
        item["id"]: item for item in manifest["tracked_tree_receipts"]
    }["eagle-i-tracked-derivatives-v1"]
    assert tree_receipt["path"] == "project/data/raw/Outage_Dataset_R1"
    assert tree_receipt["path_count"] == 52
    assert tree_receipt["total_bytes"] == 215_544_230
    assert re.fullmatch(r"[0-9a-f]{64}", tree_receipt["inventory_sha256"])
    assert tree_receipt["lineage_status"] == "ambiguous"


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


def test_full_upstream_scope_reports_dispositioned_blockers() -> None:
    validator = _load_validator()

    report = validator.validate_scope(
        root=ROOT,
        manifest_path=MANIFEST,
        scope="full-upstream",
    )

    assert report["status"] == "blocked"
    assert report["full_upstream_reproduction_established"] is False
    blocker_codes = {item["code"] for item in report["blockers"]}
    assert "restricted-input-git-tracked" not in blocker_codes
    assert "tracked-derived-lineage-unproven" in blocker_codes
    assert "earth-engine-export-receipt-missing" in blocker_codes
    assert "mutable-source-snapshot-incomplete" in blocker_codes
    assert "catalog-selection-required" in blocker_codes
    eagle_blockers = [
        item
        for item in report["blockers"]
        if item["source_id"] == "eagle_i"
    ]
    assert eagle_blockers == [
        {
            "code": "tracked-derived-lineage-unproven",
            "detail": (
                "The official EAGLE-I release is public under CC BY 4.0, but "
                "the tracked transformed and event-joined files are not tied "
                "to a recorded parent release and deterministic transform."
            ),
            "source_id": "eagle_i",
            "disposition": "ambiguous",
        }
    ]
    assert "ignored-storage-policy-mismatch" not in blocker_codes
    assert report["dispositions"] == [
        {
            "source_id": item["source_id"],
            "disposition": item["disposition"],
            "full_rerun_ready": item["full_rerun_ready"],
            "blocker_code": item.get("blocker_code"),
        }
        for item in json.loads(MANIFEST.read_text(encoding="utf-8"))[
            "upstream_boundaries"
        ]
    ]


def test_invalid_or_inconsistent_disposition_fails_closed(tmp_path: Path) -> None:
    validator = _load_validator()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["upstream_boundaries"][0]["disposition"] = "hand-wave"
    changed_manifest = tmp_path / "reproducibility_inputs_v1.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_scope(
        root=ROOT,
        manifest_path=changed_manifest,
        scope="full-upstream",
    )

    assert report["status"] == "blocked"
    assert "invalid-boundary-disposition" in {
        item["code"] for item in report["blockers"]
    }


def test_duplicate_source_boundary_fails_closed(tmp_path: Path) -> None:
    validator = _load_validator()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["upstream_boundaries"].append(manifest["upstream_boundaries"][0])
    changed_manifest = tmp_path / "reproducibility_inputs_v1.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_scope(
        root=ROOT,
        manifest_path=changed_manifest,
        scope="full-upstream",
    )

    assert report["status"] == "blocked"
    assert "source-boundary-coverage-mismatch" in {
        item["code"] for item in report["blockers"]
    }


def test_missing_boundary_action_or_evidence_fails_closed(tmp_path: Path) -> None:
    validator = _load_validator()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["upstream_boundaries"][0].pop("next_action")
    manifest["upstream_boundaries"][0]["evidence"] = []
    changed_manifest = tmp_path / "reproducibility_inputs_v1.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_scope(
        root=ROOT,
        manifest_path=changed_manifest,
        scope="full-upstream",
    )

    assert report["status"] == "blocked"
    assert "source-boundary-metadata-incomplete" in {
        item["code"] for item in report["blockers"]
    }


def test_eagle_i_tracked_tree_receipt_detects_inventory_drift(tmp_path: Path) -> None:
    validator = _load_validator()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["tracked_tree_receipts"][0]["inventory_sha256"] = "0" * 64
    changed_manifest = tmp_path / "reproducibility_inputs_v1.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_scope(
        root=ROOT,
        manifest_path=changed_manifest,
        scope="full-upstream",
    )

    assert report["status"] == "blocked"
    assert "tracked-tree-checksum-mismatch" in {
        item["code"] for item in report["blockers"]
    }
