import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CONTENT = ROOT / "project" / "nightlight-public" / "src" / "content"
PUBLIC_ARTIFACT = PUBLIC_CONTENT / "evidencePassportArtifact.js"
PUBLIC_MANIFEST = PUBLIC_CONTENT / "evidencePassportManifest.json"
PUBLIC_STUDY = PUBLIC_CONTENT / "study.js"
PRIVATE_COMPONENTS = (
    ROOT / "project" / "modeling" / "output" / "event_readiness_components_v1.csv"
)

EVENT_MAPPING = {
    "ian_charlotteharbor": "ian-charlotte",
    "ian_fortmyers": "ian-fortmyers",
    "earthquake_sanjuan": "eq-pr",
    "ida_neworleans": "ida",
    "irma_miami": "irma",
    "laura_lakecharles": "laura",
    "earthquake_hatay": "eq-hatay",
    "maria_sanjuan": "maria",
    "michael_panamacity": "michael",
}

COMPONENT_COLUMNS = {
    "observation-quality": "obs_quality_score",
    "post-event-coverage": "post_coverage_score",
    "context-coverage": "poi_score",
    "covariate-completeness": "covariate_score",
    "data-integrity": "integrity_score",
}


def canonical_sha256(path: Path) -> str:
    canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical_bytes).hexdigest()


def test_public_artifact_is_bound_to_a_versioned_reviewed_manifest():
    payload = load_public_modules()
    manifest = json.loads(PUBLIC_MANIFEST.read_text(encoding="utf-8"))
    artifact = payload["artifact"]

    assert artifact["source"]["id"] == "event-readiness-public-components-v1"
    assert artifact["source"]["sha256"] == canonical_sha256(PUBLIC_MANIFEST)
    assert manifest["source"]["privateCanonicalSha256"] == (
        "5d2f93b69913cfe93c48cc2ea81e08499502536d92f81f72a1cbe2dcfe4a3586"
    )

    manifest_passports = {
        passport["eventId"]: passport for passport in manifest["passports"]
    }
    artifact_passports = {
        passport["eventId"]: passport for passport in artifact["passports"]
    }
    assert set(artifact_passports) == set(manifest_passports)
    for event_id, reviewed in manifest_passports.items():
        published = artifact_passports[event_id]
        assert published["readinessBand"] == reviewed["readinessBand"]
        assert {
            component["id"]: component["points"]
            for component in published["components"]
        } == reviewed["componentPoints"]


def load_public_modules() -> dict:
    script = f"""
      import {{ PUBLIC_EVIDENCE_PASSPORT_ARTIFACT }} from {json.dumps(PUBLIC_ARTIFACT.as_uri())};
      import {{ EVENTS }} from {json.dumps(PUBLIC_STUDY.as_uri())};
      console.log(JSON.stringify({{ artifact: PUBLIC_EVIDENCE_PASSPORT_ARTIFACT, events: EVENTS }}));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def test_optional_private_components_match_the_reviewed_public_manifest():
    if not PRIVATE_COMPONENTS.exists():
        pytest.skip("private readiness source is unavailable in this clone")

    manifest = json.loads(PUBLIC_MANIFEST.read_text(encoding="utf-8"))
    with PRIVATE_COMPONENTS.open(newline="", encoding="utf-8") as handle:
        private_rows = {row["event_id"]: row for row in csv.DictReader(handle)}

    assert manifest["source"]["privateCanonicalSha256"] == canonical_sha256(
        PRIVATE_COMPONENTS
    )
    assert set(private_rows) == set(EVENT_MAPPING) | {"dorian_freeport"}

    passports = {
        passport["eventId"]: passport for passport in manifest["passports"]
    }
    assert set(passports) == set(EVENT_MAPPING.values())
    assert "dorian-freeport" not in passports

    for private_event_id, public_event_id in EVENT_MAPPING.items():
        private_row = private_rows[private_event_id]
        passport = passports[public_event_id]
        assert passport["componentPoints"] == {
            component_id: int(private_row[column])
            for component_id, column in COMPONENT_COLUMNS.items()
        }
        assert passport["readinessBand"] == private_row["readiness_band"]


def test_public_passport_cohort_is_a_strict_subset_with_no_reconstructable_keys():
    payload = load_public_modules()
    artifact = payload["artifact"]
    event_ids = {event["id"] for event in payload["events"]}
    passport_ids = {passport["eventId"] for passport in artifact["passports"]}

    assert len(event_ids) == 25
    assert len(passport_ids) == 9
    assert len(event_ids - passport_ids) == 16
    assert passport_ids < event_ids

    prohibited_keys = {
        "eventCount",
        "observedRate",
        "highCensoringShare",
        "poiCount",
        "totalScore",
        "incrementImpactLabel",
        "recommendedRole",
        "facility",
        "coordinates",
        "timeSeries",
        "localPath",
    }

    def walk(value):
        if isinstance(value, dict):
            assert prohibited_keys.isdisjoint(value)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(artifact)
    serialized = json.dumps(artifact)
    assert "project/modeling" not in serialized
    assert "event_readiness_components_v1.csv" not in serialized
    assert not any(private_event_id in serialized for private_event_id in EVENT_MAPPING)
