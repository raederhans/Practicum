import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "recovery_label_source_feasibility_v1.json"
)
PILOT_PATH = (
    PROJECT_ROOT / "data" / "manifests" / "recovery_label_pilot_v1.json"
)
MODULE_PATH = PROJECT_ROOT / "modeling" / "support" / "source_feasibility.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_source_feasibility", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_blocked_source_gate_prevents_any_label_pilot_rows():
    module = _load_module()
    feasibility = _read(FEASIBILITY_PATH)
    pilot = _read(PILOT_PATH)

    evaluated = module.evaluate_label_pilot_gate(feasibility)
    assert evaluated["decision"] == "blocked"
    assert pilot["status"] == "evidence-backed-blocked"
    assert pilot["upstream_gate_decision"] == evaluated["decision"]
    assert pilot["selected_events"] == []
    assert pilot["labels"] == []
    assert pilot["qa"]["event_count"] == 0
    assert pilot["qa"]["label_count"] == 0


def test_pilot_blockers_and_handoff_match_the_admission_source():
    feasibility = _read(FEASIBILITY_PATH)
    pilot = _read(PILOT_PATH)
    gate = feasibility["label_pilot_gate"]

    assert pilot["blocker_codes"] == gate["blocker_codes"]
    assert pilot["executable_handoff"] == gate["executable_handoff"]
    assert "every required" in pilot["admission_invariant"].lower()


def test_blocked_pilot_uses_no_mock_training_publication_or_headline_changes():
    pilot = _read(PILOT_PATH)
    qa = pilot["qa"]

    for invariant in (
        "mock_or_synthetic_labels_used",
        "ground_truth_rows_read",
        "external_data_downloaded",
        "credential_content_read",
        "forecast_or_probability_trained",
        "forecast_or_probability_published",
        "headline_scientific_metric_changed",
        "public_app_modified",
        "dashboard_modified",
    ):
        assert qa[invariant] is False
