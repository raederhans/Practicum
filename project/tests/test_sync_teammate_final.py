from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "project" / "modeling" / "support" / "sync_teammate_final.py"
MANIFEST_DIR = ROOT / "project" / "data" / "manifests"
DONOR_COMMIT = "1f63e190ce280852d68945dbfce486075adda69b"


def load_module():
    if not SCRIPT.is_file():
        raise AssertionError(f"missing implementation: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("sync_teammate_final", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True
    ).stdout


class ManifestContractTests(unittest.TestCase):
    def test_sync_inventory_is_explicitly_an_initial_import_receipt(self):
        inventory = json.loads(
            (MANIFEST_DIR / "sync_inventory_v1.json").read_text(encoding="utf-8")
        )

        self.assertEqual("initial-import-receipt", inventory["artifact_role"])
        self.assertEqual("2026-08-05T04:55:12Z", inventory["captured_at"])
        self.assertEqual("immediate-post-import", inventory["state_scope"])
        self.assertTrue(inventory["current_tree_may_diverge"])
        self.assertIn("not a current-tree assertion", inventory["scope_note"])
        self.assertEqual(343, inventory["summary"]["skip_identical"])

    def test_sync_plan_is_commit_pinned_and_narrow(self):
        plan = json.loads((MANIFEST_DIR / "sync_plan_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(DONOR_COMMIT, plan["donor_commit"])
        self.assertEqual("teammate/main", plan["reachability_ref"])
        self.assertEqual(
            [
                "project/nightlight-dashboard",
                "project/data/result/stage2",
                "project/data/result/stage3",
            ],
            plan["slices"]["roots"],
        )
        self.assertIn("project/script/stage3_events.py", plan["slices"]["files"])
        self.assertTrue(
            {
                "project/script/run_modelD_loeo_25events.py",
                "project/script/regen_modelD_prob_maps.py",
                "project/script/make_modelD_loeo_heatmap.py",
            }.issubset(plan["slices"]["files"])
        )
        self.assertNotIn("project/script", plan["slices"]["roots"])
        self.assertFalse(
            any(
                path == "project/data/raw" or path.startswith("project/data/raw/")
                for path in [*plan["slices"]["roots"], *plan["slices"]["files"]]
            )
        )
        protected = set(plan["protected_prefixes"])
        self.assertTrue(
            {
                "README.md",
                ".gitignore",
                "project/modeling",
                "project/modeling_tracking",
                "project/modeling_report",
                "notebooks",
                "document",
                "extra data",
                "POI",
                "run_wsl_modeling.ps1",
            }.issubset(protected)
        )

    def test_source_manifest_records_all_required_official_sources(self):
        manifest = json.loads(
            (MANIFEST_DIR / "source_manifest_v1.json").read_text(encoding="utf-8")
        )
        sources = {item["id"]: item for item in manifest["sources"]}
        self.assertEqual(
            {
                "nasa_vnp46a2",
                "eagle_i",
                "osm_overpass",
                "census_tiger_zcta_county",
                "census_acs_zcta_2022",
                "noaa_ibtracs",
                "noaa_hurdat2_atlantic",
                "miami_dade_open_data",
                "nlcd",
                "worldpop",
            },
            set(sources),
        )
        required = {
            "official_urls",
            "auth",
            "license",
            "version",
            "redistribution",
            "reproducibility",
            "status",
            "local_targets",
        }
        for source in sources.values():
            self.assertTrue(required.issubset(source), source["id"])
            self.assertTrue(source["official_urls"], source["id"])
        self.assertEqual("partner-restricted", sources["eagle_i"]["status"])

    def test_canonical_results_only_asserts_evidenced_values(self):
        results = json.loads(
            (MANIFEST_DIR / "canonical_results_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(25, results["scope"]["stage2_events"])
        self.assertEqual(22, results["scope"]["stage3_events"])
        self.assertEqual(15, results["scope"]["stage3_us_states"])
        self.assertEqual(2, results["scope"]["puerto_rico_events"])
        self.assertEqual(1, results["scope"]["turkey_events"])
        self.assertEqual(
            "17 jurisdictions across the U.S. and Turkey",
            results["scope"]["publication_phrase"],
        )
        regression = results["stage3_regression"]
        self.assertEqual("publishable-with-caveats", regression["status"])
        self.assertEqual("in-sample-fit", regression["metric_kind"])
        self.assertEqual(0.7603, regression["r_squared"])
        self.assertEqual(0.7543, regression["adjusted_r_squared"])
        self.assertEqual(977, regression["n"])
        self.assertEqual(22, regression["event_clusters"])
        self.assertEqual(
            0.551,
            regression["exploratory_facility_density_ratio"]["value"],
        )
        self.assertEqual(
            "descriptive-only",
            regression["exploratory_facility_density_ratio"]["status"],
        )
        self.assertIn(
            "project/data/result/stage3/regression_results_modelD_extra.json",
            regression["sources"],
        )
        donor = results["retired_donor_baseline"]
        self.assertEqual("retired", donor["status"])
        self.assertEqual(0.7472, donor["r_squared"])
        self.assertEqual(0.7408, donor["adjusted_r_squared"])
        self.assertEqual(0.63, donor["facility_density_ratio"])
        self.assertEqual("rf", donor["probability_variant"])
        self.assertEqual(1, donor["probability_tif_band"])
        self.assertEqual("EPSG:3857", donor["area_crs"])
        self.assertEqual("approximately", results["model_d_rf_loeo_mean_auc"]["qualifier"])
        self.assertAlmostEqual(0.704040, results["model_d_rf_loeo_mean_auc"]["value"], places=6)
        self.assertEqual("ensemble", results["probability_variant"])
        self.assertEqual(3, results["probability_tif_band"])
        self.assertEqual("EPSG:5070", results["area_crs"])
        self.assertEqual("3.12.10", results["runtime_contract"]["python"])
        self.assertEqual(4, len(results["formal_artifacts"]))
        for artifact in results["formal_artifacts"].values():
            self.assertEqual(64, len(artifact["sha256"]))
        self.assertIn("not causal", " ".join(results["publication_guardrails"]).lower())


class SyncBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.sync = load_module()

    def small_plan(self, files: list[str]) -> dict:
        return {
            "schema_version": 1,
            "donor_commit": DONOR_COMMIT,
            "reachability_ref": "teammate/main",
            "slices": {"roots": [], "files": files},
            "protected_prefixes": ["README.md", "project/modeling"],
        }

    def test_rejects_noncanonical_and_duplicate_paths(self):
        bad_paths = [
            "../escape",
            "/absolute",
            "C:/absolute",
            "project\\bad",
            "a//b",
            "./a",
            "a/../b",
        ]
        for bad in bad_paths:
            with self.subTest(path=bad):
                with self.assertRaises(self.sync.SyncError):
                    self.sync.validate_relative_path(bad)
        with self.assertRaises(self.sync.SyncError):
            self.sync.validate_plan(
                self.small_plan(
                    [
                        ".github/workflows/deploy-dashboard.yml",
                        ".github/workflows/deploy-dashboard.yml",
                    ]
                )
            )

    def test_check_writes_inventory_only_and_records_blob_evidence(self):
        rel = ".github/workflows/deploy-dashboard.yml"
        source = git_bytes("show", f"{DONOR_COMMIT}:{rel}")
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)
            inventory_path = target / "sync_inventory_v1.json"
            result = self.sync.synchronize(
                git_root=ROOT,
                target_root=target,
                plan=self.small_plan([rel]),
                apply=False,
                inventory_path=inventory_path,
            )
            self.assertFalse((target / rel).exists())
            self.assertTrue(inventory_path.is_file())
            entry = result["files"][0]
            self.assertEqual("copy", entry["action"])
            self.assertEqual(DONOR_COMMIT, entry["donor_commit"])
            self.assertEqual(len(source), entry["bytes"])
            self.assertEqual(hashlib.sha256(source).hexdigest(), entry["sha256"])
            self.assertRegex(entry["git_oid"], r"^[0-9a-f]{40,64}$")
            self.assertEqual("explicit-files", entry["slice"])
            created = sorted(
                p.relative_to(target).as_posix()
                for p in target.rglob("*")
                if p.is_file()
            )
            self.assertEqual(["sync_inventory_v1.json"], created)

    def test_apply_copies_missing_and_skips_identical(self):
        rel = ".github/workflows/deploy-dashboard.yml"
        source = git_bytes("show", f"{DONOR_COMMIT}:{rel}")
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)
            inventory_path = target / "sync_inventory_v1.json"
            first = self.sync.synchronize(
                git_root=ROOT,
                target_root=target,
                plan=self.small_plan([rel]),
                apply=True,
                inventory_path=inventory_path,
            )
            self.assertEqual(source, (target / rel).read_bytes())
            self.assertEqual("copied", first["files"][0]["status"])
            second = self.sync.synchronize(
                git_root=ROOT,
                target_root=target,
                plan=self.small_plan([rel]),
                apply=True,
                inventory_path=inventory_path,
            )
            self.assertEqual("skip-identical", second["files"][0]["status"])

    def test_conflict_fails_closed_before_any_copy(self):
        first = ".github/workflows/deploy-dashboard.yml"
        second = "project/script/stage3_events.py"
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)
            conflict = target / second
            conflict.parent.mkdir(parents=True)
            conflict.write_bytes(b"local content must survive")
            with self.assertRaises(self.sync.SyncConflictError):
                self.sync.synchronize(
                    git_root=ROOT,
                    target_root=target,
                    plan=self.small_plan([first, second]),
                    apply=True,
                    inventory_path=target / "sync_inventory_v1.json",
                )
            self.assertFalse((target / first).exists())
            self.assertEqual(b"local content must survive", conflict.read_bytes())


if __name__ == "__main__":
    unittest.main()
