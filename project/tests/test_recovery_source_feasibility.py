import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "recovery_label_source_feasibility_v1.json"
)
MODULE_PATH = PROJECT_ROOT / "modeling" / "support" / "source_feasibility.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_source_feasibility", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_source_inventory_is_complete_and_fail_closed():
    module = _load_module()
    manifest = _manifest()

    module.validate_manifest(manifest)
    sources = {source["id"]: source for source in manifest["sources"]}
    assert set(sources) == module.REQUIRED_SOURCE_IDS
    assert manifest["status"] == "evidence-backed-blocked"
    assert manifest["evidence_policy"]["external_data_downloaded"] is False
    assert manifest["evidence_policy"]["credential_content_read"] is False
    assert manifest["evidence_policy"]["raw_or_cache_bytes_added_to_git"] is False


def test_source_publicity_does_not_substitute_for_label_rebuildability():
    manifest = _manifest()
    sources = {source["id"]: source for source in manifest["sources"]}

    eagle = sources["eagle_i"]
    assert eagle["label_eligibility"]["independent_ground_truth"] is True
    assert eagle["label_eligibility"]["status"] == "candidate-not-admitted"
    assert eagle["missingness"]["zero_distinguishable_from_missing"] is False
    assert eagle["rebuildability"]["ready"] is False
    assert "tracked-derived-lineage-unproven" in eagle["rebuildability"]["blockers"]

    vnp = sources["nasa_vnp46a2"]
    assert vnp["label_eligibility"]["status"] == "proxy-only"
    assert vnp["label_eligibility"]["independent_ground_truth"] is False
    assert vnp["access"]["probe_status"].endswith("export-not-attempted")


def test_utility_alternatives_are_not_misrepresented_as_event_labels():
    manifest = _manifest()
    sources = {source["id"]: source for source in manifest["sources"]}

    assert sources["doe_oe417"]["label_eligibility"]["status"] == (
        "event-anchor-only-not-ground-truth-panel"
    )
    assert sources["eia_861"]["label_eligibility"]["status"] == (
        "denominator-context-only"
    )
    assert sources["direct_utility_outage_maps"]["publication_rights"][
        "redistribution"
    ] == "unknown-until-source-specific-review"


def test_existing_authorized_sources_remain_context_not_ground_truth():
    manifest = _manifest()
    context_sources = manifest["existing_authorized_context_sources"]

    assert {entry["source_id"] for entry in context_sources} == {
        "census_tiger_zcta_county",
        "census_acs_zcta_2022",
        "noaa_hurdat2_atlantic",
        "osm_overpass",
        "nlcd",
        "worldpop",
    }
    assert all(entry["label_status"] != "ground-truth" for entry in context_sources)


def test_label_pilot_gate_exposes_exact_blockers_and_executable_handoff():
    module = _load_module()
    manifest = _manifest()

    evaluated = module.evaluate_label_pilot_gate(manifest)
    assert evaluated["decision"] == "blocked"
    assert {
        "rights_for_selected_parent_and_derivatives_verified",
        "exact_parent_and_transform_receipts_complete",
        "event_time_pinned",
        "event_denominator_pinned",
        "independent_ground_truth_rebuildable",
        "missingness_zero_and_unavailable_distinguishable",
    }.issubset(evaluated["failed_gates"])
    gate = manifest["label_pilot_gate"]
    assert "zero-versus-collection-gap-undistinguished" in gate["blocker_codes"]
    assert len(gate["executable_handoff"]) == 5
