import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "project" / "nightlight-dashboard"
PAGES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-dashboard.yml"


def read(relative_path: str) -> str:
    return (DASHBOARD / relative_path).read_text(encoding="utf-8")


def test_dashboard_never_uses_generated_data_when_exports_are_missing():
    loader = read("src/data/loader.js")
    event_data = read("src/data/events.js")
    time_series = read("src/data/timeseries.js")

    assert "generateMock" not in loader
    assert "generateRecoverySeries" not in loader
    assert "return null" not in loader
    assert "DataLoadError" in loader
    assert "validateFeatureCollection" in loader
    assert "validateTimeSeries" in loader
    assert "generateMockProbabilityGeoJSON" not in event_data
    assert "generateFacilityGeoJSON" not in event_data
    assert "generateRecoverySeries" not in time_series


def test_dashboard_exposes_charts_route_and_has_no_root_background_asset_reference():
    router = read("src/router/index.js")
    vue_and_css = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (DASHBOARD / "src").rglob("*")
        if path.suffix in {".vue", ".js", ".css"}
    )

    assert "path: '/charts'" in router
    assert "ChartsView.vue" in router
    assert "/earth-night.jpg" not in vue_and_css


def test_dashboard_surfaces_data_unavailable_states():
    map_view = read("src/views/MapView.vue")
    recovery_chart = read("src/components/RecoveryChart.vue")
    charts_view = read("src/views/ChartsView.vue")

    for source in (map_view, recovery_chart, charts_view):
        assert "Data unavailable" in source
        assert "dataError" in source


def test_charts_only_offer_models_with_real_loeo_rows():
    charts = read("src/views/ChartsView.vue")
    results = json.loads(read("public/data/results_summary.json"))

    assert "availableModelOptions" in charts
    assert "return results.value.loeo\n" not in charts
    assert set(results["loeo_by_model"]) == {"A", "D"}
    assert len(results["loeo_by_model"]["D"]) == 25
    rf_mean = sum(row["rf_auc"] for row in results["loeo_by_model"]["D"]) / 25
    assert abs(rf_mean - 0.7040403776172799) < 1e-12


def test_model_comparison_copy_matches_exported_aggregates():
    charts = read("src/views/ChartsView.vue")
    slides = read("src/views/SlidesView.vue")
    results = json.loads(read("public/data/results_summary.json"))

    assert f'<div class="auc-cell__val">{results["model_comparison"]["model_b"]["mean_auc"]:.3f}</div>' in slides
    assert f'<div class="auc-cell__val">{results["model_comparison"]["model_c"]["mean_auc"]:.3f}</div>' in slides
    expected_gain = results["model_comparison"]["model_a"]["mean_auc"] - results["model_comparison"]["model_d"]["mean_auc"]
    assert f"+{expected_gain:.3f} AUC" in charts


def test_key_results_copy_distinguishes_full_and_pure_ntl_models():
    charts = read("src/views/ChartsView.vue")

    assert "v-if=\"activeModel === 'A'\"" in charts
    assert "v-else-if=\"activeModel === 'D'\"" in charts
    assert "Full spatial model" in charts
    assert "Pure NTL model" in charts


def test_canonical_project_scope_and_stage3_metrics_are_visible():
    prose = "\n".join(
        read(relative_path)
        for relative_path in (
            "src/views/HomeView.vue",
            "src/views/SlidesView.vue",
            "src/views/DocsView.vue",
            "src/views/DocsDetailView.vue",
        )
    )

    assert "Stage 2" in prose and "25 events" in prose
    assert "Stage 3" in prose and "22 events" in prose and "15 U.S. states" in prose
    assert "N = 977" in prose
    assert "R² = 0.7603" in prose
    assert "adjusted R² = 0.7543" in prose
    assert "55.1%" in prose
    assert "in-sample fit" in prose
    assert "not a causal estimate" in prose
    assert "R² = 0.7472" not in prose
    assert "adjusted R² = 0.7408" not in prose
    assert "63%" not in prose
    assert "R² = 0.475" not in prose
    assert "only 38%" not in prose
    assert "19 U.S. states" not in prose
    assert "17 jurisdictions across the U.S. and Turkey" in prose
    assert "const stateCount = 19" not in prose


def test_personal_dashboard_leads_with_owner_and_credits_the_collaboration():
    home = read("src/views/HomeView.vue")
    slides = read("src/views/SlidesView.vue")

    assert "Qiushi Yu" in home
    assert "Personal continuation" in home
    assert "Original practicum with Zhiyuan Zhao" in home
    assert "Personal Project Continuation" in slides
    assert "Original practicum with Zhiyuan Zhao" in slides


def test_dashboard_states_public_upstream_lineage_and_local_release_boundaries():
    prose = "\n".join(
        read(relative_path)
        for relative_path in (
            "src/views/DocsView.vue",
            "src/views/DocsDetailView.vue",
        )
    )

    assert "official upstream EAGLE-I release is public" in prose
    assert "CC BY 4.0" in prose
    assert "repository derivative lineage is unproven" in prose
    assert "partner-restricted" not in prose
    assert "partner access required" not in prose
    assert "authorized local" not in prose
    assert "not redistributed" in prose
    assert "has not been publicly deployed" in prose
    assert "All data sources are publicly available" not in prose
    assert "DOE EAGLE-I portal (public)" not in prose
    assert "is deployed via GitHub Actions" not in prose


def test_dashboard_data_exports_the_canonical_stage3_contract():
    results = json.loads(read("public/data/results_summary.json"))
    stage3 = results["stage3"]

    assert stage3["status"] == "publishable-with-caveats"
    assert stage3["unit_of_analysis"] == "ZIP-event"
    assert stage3["events"] == 22
    assert stage3["states"] == 15
    assert stage3["n"] == 977
    assert stage3["r_squared"] == 0.7603
    assert stage3["adjusted_r_squared"] == 0.7543
    assert stage3["exploratory_facility_density_ratio"] == 0.551
    assert any("not a causal estimate" in item for item in stage3["guardrails"])


def test_pages_workflow_runs_dashboard_tests_before_build():
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")

    test_step = workflow.index("run: npm test")
    build_step = workflow.index("run: npm run build")
    assert test_step < build_step


def test_pages_workflow_scopes_permissions_to_each_job():
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")
    build_job = workflow[workflow.index("  build:"):workflow.index("\n  deploy:")]
    deploy_job = workflow[workflow.index("  deploy:"):]

    assert "\npermissions:" not in workflow
    assert "\n    permissions:\n      contents: read\n" in build_job
    assert "pages: write" not in build_job
    assert "id-token: write" not in build_job
    assert "\n    permissions:\n      pages: write\n      id-token: write\n" in deploy_job
    assert "contents: read" not in deploy_job
    assert workflow.count("pages: write") == 1
    assert workflow.count("id-token: write") == 1


def test_pages_workflow_pins_official_actions_to_reviewed_commits():
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")
    reviewed_actions = {
        "actions/checkout": ("11d5960a326750d5838078e36cf38b85af677262", "v4.4.0"),
        "actions/setup-node": ("49933ea5288caeca8642d1e84afbd3f7d6820020", "v4.4.0"),
        "actions/upload-pages-artifact": ("56afc609e74202658d3ffba0e8f6dda462b719fa", "v3.0.1"),
        "actions/deploy-pages": ("d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e", "v4.0.5"),
    }

    for action, (commit, version) in reviewed_actions.items():
        assert f"uses: {action}@{commit} # {version}" in workflow


def test_dashboard_pins_reviewed_vite_and_vitest_security_lines():
    package = json.loads(read("package.json"))

    assert package["devDependencies"]["vite"] == "6.4.3"
    assert package["devDependencies"]["vitest"] == "3.2.7"
