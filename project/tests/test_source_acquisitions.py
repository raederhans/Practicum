from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "project" / "data" / "manifests" / "source_manifest_v1.json"


class SourceAcquisitionContractTests(unittest.TestCase):
    def test_census_2020_archives_have_reproducible_receipts(self) -> None:
        manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        sources = {item["id"]: item for item in manifest["sources"]}
        census = sources["census_tiger_zcta_county"]
        receipts = {item["filename"]: item for item in census["acquisitions"]}

        expected = {
            "tl_2020_us_zcta520.zip": (
                "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/"
                "tl_2020_us_zcta520.zip",
                527_995_578,
            ),
            "tl_2020_us_county.zip": (
                "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/"
                "tl_2020_us_county.zip",
                80_644_766,
            ),
        }
        self.assertEqual(set(expected), set(receipts))
        for filename, (url, byte_count) in expected.items():
            receipt = receipts[filename]
            self.assertEqual(url, receipt["url"])
            self.assertEqual(byte_count, receipt["bytes"])
            self.assertRegex(receipt["sha256"], re.compile(r"^[0-9a-f]{64}$"))
            self.assertRegex(receipt["retrieved_at"], re.compile(r"^2026-08-05T"))
            self.assertEqual("ignored-local-cache", receipt["storage"])
            self.assertEqual("verified", receipt["status"])

        validation = census["validation"]
        self.assertEqual("EPSG:4269", validation["crs"])
        self.assertEqual(33_791, validation["zcta_rows"])
        self.assertEqual(33_791, validation["zcta_valid_geometry_rows"])
        self.assertEqual(3_234, validation["county_rows"])
        self.assertEqual(3_234, validation["county_valid_geometry_rows"])
        self.assertEqual("ZCTA5CE20", validation["zcta_id_field"])
        self.assertEqual("GEOID", validation["county_id_field"])
        self.assertEqual("geopandas-read-and-geometry-check", validation["method"])

    def test_acs_hurdat2_and_osm_receipts_capture_real_acquisitions(self) -> None:
        manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        sources = {item["id"]: item for item in manifest["sources"]}

        acs = sources["census_acs_zcta_2022"]
        self.assertEqual("public-download-verified", acs["status"])
        self.assertEqual(33_774, acs["acquisition"]["rows"])
        self.assertEqual(
            "2cae6c00da19cb97ce0ccbd2600f5e3bfdd85cdfa1a310339cce0421537c1df3",
            acs["acquisition"]["output_sha256"],
        )
        self.assertEqual("860Z200US", acs["acquisition"]["geo_id_prefix"])
        self.assertEqual(2, len(acs["acquisition"]["responses"]))

        hurdat = sources["noaa_hurdat2_atlantic"]
        self.assertEqual("public-download-verified", hurdat["status"])
        self.assertEqual(55_605, hurdat["acquisition"]["rows"])
        self.assertEqual(2_004, hurdat["acquisition"]["storms"])
        self.assertEqual(
            "1b9b0c7beed5b4505838658b1d30e159fc84330c60891a58cfcf43ae55c37202",
            hurdat["acquisition"]["source_sha256"],
        )
        self.assertEqual(
            "09712f2926b8026f79b09bf2457e4397a9495e2148c0c7170e9813c0287d3cce",
            hurdat["acquisition"]["output_sha256"],
        )

        osm = sources["osm_overpass"]
        live = osm["validation_acquisition"]
        self.assertEqual("zeta_atlanta", live["event_id"])
        self.assertEqual(399, live["row_count"])
        self.assertEqual(404, live["donor_row_count"])
        self.assertEqual(
            "af791973010fd5bb2a3e0eb4dc8f37c19c444c9b1be6fbba287d77ed7cfdc8eb",
            live["sha256"],
        )
        self.assertIn("Practicum-reproducibility", live["user_agent"])

        ibtracs = sources["noaa_ibtracs"]
        self.assertEqual("noaa_hurdat2_atlantic", ibtracs["run_alternative"])

    def test_public_eagle_i_release_keeps_local_derivative_lineage_unresolved(self) -> None:
        manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        sources = {item["id"]: item for item in manifest["sources"]}
        eagle_i = sources["eagle_i"]

        tracked_path = "project/data/raw/Outage_Dataset_R1"
        self.assertEqual([tracked_path], eagle_i["local_targets"])
        self.assertTrue((ROOT / tracked_path).is_dir())
        self.assertEqual(
            [
                {
                    "path": "project/data/result/stage3",
                    "publication_status": "lineage-review-required",
                }
            ],
            eagle_i["derived_targets"],
        )
        self.assertEqual(
            "public-release-local-derivatives-lineage-unproven",
            eagle_i["status"],
        )
        self.assertEqual(
            "public-source-permitted-with-CC-BY-attribution; "
            "transformed-or-joined-files-require-lineage-review",
            eagle_i["redistribution"],
        )
        release = eagle_i["official_release"]
        self.assertEqual(24_237_376, release["article_id"])
        self.assertEqual(4, release["version"])
        self.assertEqual("CC BY 4.0", release["license"])
        self.assertEqual(10, len(release["files_used_by_repository_year_range"]))


if __name__ == "__main__":
    unittest.main()
