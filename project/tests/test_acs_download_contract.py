from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "project" / "script" / "stage3_acs_download.py"


def load_module():
    if not SCRIPT.is_file():
        raise AssertionError(f"missing implementation: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("stage3_acs_download", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AcsDownloadContractTests(unittest.TestCase):
    def setUp(self):
        self.acs = load_module()

    def test_download_sorts_zctas_blanks_sentinels_and_hashes_outputs(self):
        response = json.dumps(
            [
                ["NAME", "B01003_001E", "B19013_001E", "zip code tabulation area"],
                ["ZCTA 12", "100", "-666666666", "12"],
                ["ZCTA 90210", "-999999999", "85000", "90210"],
                ["ZCTA 00001", "5", "12000", "00001"],
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        seen_urls: list[str] = []

        def opener(url: str, timeout: float):
            self.assertEqual(60.0, timeout)
            seen_urls.append(url)
            return io.BytesIO(response)

        with tempfile.TemporaryDirectory() as td:
            result = self.acs.download_acs_zcta(
                output_dir=Path(td),
                year=2022,
                opener=opener,
                retrieved_at=lambda: "2026-08-05T05:00:00Z",
            )
            csv_path = Path(td) / "acs_zcta_2022.csv"
            manifest_path = Path(td) / "acs_zcta_2022.manifest.json"
            expected_csv = (
                "ZCTA5CE20,total_pop,median_income,NAME\r\n"
                "00001,5,12000,ZCTA 00001\r\n"
                "00012,100,,ZCTA 12\r\n"
                "90210,,85000,ZCTA 90210\r\n"
            ).encode("utf-8")
            self.assertEqual(expected_csv, csv_path.read_bytes())
            self.assertEqual(csv_path, result["csv_path"])
            self.assertEqual(manifest_path, result["manifest_path"])

            query = parse_qs(urlparse(seen_urls[0]).query)
            self.assertEqual(["NAME,B01003_001E,B19013_001E"], query["get"])
            self.assertEqual(["zip code tabulation area:*"], query["for"])
            self.assertIn("/data/2022/acs/acs5", seen_urls[0])

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("Census ACS 5-year ZCTA", manifest["dataset"])
            self.assertEqual(2022, manifest["year"])
            self.assertEqual(seen_urls[0], manifest["api_url"])
            self.assertEqual(
                {
                    "NAME": "NAME",
                    "total_pop": "B01003_001E",
                    "median_income": "B19013_001E",
                },
                manifest["variables"],
            )
            self.assertEqual("2026-08-05T05:00:00Z", manifest["retrieved_at"])
            self.assertEqual(hashlib.sha256(response).hexdigest(), manifest["response_sha256"])
            self.assertEqual(hashlib.sha256(expected_csv).hexdigest(), manifest["output_sha256"])
            self.assertEqual(3, manifest["rows"])
            self.assertTrue(manifest["license"])
            self.assertIn("U.S. Census Bureau", manifest["citation"])

    def test_http_error_fails_without_output_files(self):
        def opener(_url: str, timeout: float):
            self.assertEqual(60.0, timeout)
            raise urllib.error.HTTPError(
                url="https://api.census.gov/",
                code=503,
                msg="service unavailable",
                hdrs=None,
                fp=None,
            )

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(self.acs.DownloadError):
                self.acs.download_acs_zcta(Path(td), opener=opener)
            self.assertEqual([], list(Path(td).iterdir()))

    def test_summary_file_mode_joins_tables_and_records_each_response(self):
        population = (
            "GEO_ID|B01003_E001|B01003_M001\n"
            "860Z200US90210|20|1\n"
            "0400000US06|999|1\n"
            "860Z200US00001|5|1\n"
        ).encode("utf-8")
        income = (
            "GEO_ID|B19013_E001|B19013_M001\n"
            "860Z200US00001|12000|100\n"
            "860Z200US90210|-666666666|-222222222\n"
        ).encode("utf-8")
        seen: list[str] = []

        def opener(url: str, timeout: float):
            self.assertEqual(60.0, timeout)
            seen.append(url)
            return io.BytesIO(population if "b01003" in url else income)

        with tempfile.TemporaryDirectory() as td:
            self.acs.download_acs_zcta(
                Path(td),
                source="summary-file",
                opener=opener,
                retrieved_at=lambda: "2026-08-05T06:00:00Z",
            )
            expected_csv = (
                "ZCTA5CE20,total_pop,median_income,NAME\r\n"
                "00001,5,12000,ZCTA5 00001\r\n"
                "90210,20,,ZCTA5 90210\r\n"
            ).encode("utf-8")
            output = Path(td) / "acs_zcta_2022.csv"
            self.assertEqual(expected_csv, output.read_bytes())
            manifest = json.loads(
                (Path(td) / "acs_zcta_2022.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("summary-file", manifest["source"])
            self.assertEqual(2, len(manifest["responses"]))
            by_variable = {item["variable"]: item for item in manifest["responses"]}
            self.assertEqual("B01003_E001", by_variable["total_pop"]["field"])
            self.assertEqual("B19013_E001", by_variable["median_income"]["field"])
            self.assertEqual(hashlib.sha256(population).hexdigest(), by_variable["total_pop"]["sha256"])
            self.assertEqual(hashlib.sha256(income).hexdigest(), by_variable["median_income"]["sha256"])
            self.assertEqual(set(seen), {item["url"] for item in manifest["responses"]})
            self.assertIn("NAME", manifest["derived_fields"])
            self.assertEqual(
                {
                    "level": "zip code tabulation area",
                    "geo_id_prefix": "860Z200US",
                },
                manifest["geography"],
            )
            self.assertEqual(hashlib.sha256(expected_csv).hexdigest(), manifest["output_sha256"])
            self.assertEqual(2, manifest["rows"])
            self.assertIn("table-based summary files", manifest["citation"])

    def test_summary_file_rejects_missing_geo_id(self):
        population = b"GEO_ID|B01003_E001|B01003_M001\n|5|1\n"
        income = b"GEO_ID|B19013_E001|B19013_M001\n860Z200US00001|12000|1\n"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(self.acs.SchemaError):
                self.acs.download_acs_zcta(
                    Path(td),
                    source="summary-file",
                    opener=lambda url, timeout: io.BytesIO(
                        population if "b01003" in url else income
                    ),
                )
            self.assertEqual([], list(Path(td).iterdir()))

    def test_summary_file_rejects_duplicate_zcta(self):
        population = (
            "GEO_ID|B01003_E001|B01003_M001\n"
            "860Z200US00001|5|1\n"
            "860Z200US00001|6|1\n"
        ).encode("utf-8")
        income = b"GEO_ID|B19013_E001|B19013_M001\n860Z200US00001|12000|1\n"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(self.acs.SchemaError):
                self.acs.download_acs_zcta(
                    Path(td),
                    source="summary-file",
                    opener=lambda url, timeout: io.BytesIO(
                        population if "b01003" in url else income
                    ),
                )

    def test_summary_file_rejects_mismatched_zcta_sets(self):
        population = b"GEO_ID|B01003_E001|B01003_M001\n860Z200US00001|5|1\n"
        income = b"GEO_ID|B19013_E001|B19013_M001\n860Z200US90210|12000|1\n"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(self.acs.SchemaError):
                self.acs.download_acs_zcta(
                    Path(td),
                    source="summary-file",
                    opener=lambda url, timeout: io.BytesIO(
                        population if "b01003" in url else income
                    ),
                )

    def test_summary_file_rejects_response_without_zctas(self):
        population = b"GEO_ID|B01003_E001|B01003_M001\n0400000US06|5|1\n"
        income = b"GEO_ID|B19013_E001|B19013_M001\n0400000US06|12000|1\n"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(self.acs.SchemaError):
                self.acs.download_acs_zcta(
                    Path(td),
                    source="summary-file",
                    opener=lambda url, timeout: io.BytesIO(
                        population if "b01003" in url else income
                    ),
                )

    def test_summary_file_rejects_unverified_year_before_fetch(self):
        calls: list[str] = []

        def opener(url: str, timeout: float):
            calls.append(url)
            return io.BytesIO(b"")

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self.acs.download_acs_zcta(
                    Path(td), year=2021, source="summary-file", opener=opener
                )
        self.assertEqual([], calls)

    def test_schema_error_fails_without_output_files(self):
        malformed = json.dumps(
            [["NAME", "B01003_001E", "zip code tabulation area"], ["ZCTA 1", "4", "1"]]
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(self.acs.SchemaError):
                self.acs.download_acs_zcta(
                    Path(td), opener=lambda _url, timeout: io.BytesIO(malformed)
                )
            self.assertEqual([], list(Path(td).iterdir()))

    def test_main_returns_nonzero_for_download_failure(self):
        with mock.patch.object(
            self.acs, "download_acs_zcta", side_effect=self.acs.DownloadError("offline")
        ):
            with redirect_stdout(io.StringIO()):
                self.assertNotEqual(0, self.acs.main(["--output-dir", "unused"]))

    def test_cli_routes_explicit_summary_file_source(self):
        result_paths = {
            "csv_path": Path("unused/acs_zcta_2022.csv"),
            "manifest_path": Path("unused/acs_zcta_2022.manifest.json"),
        }
        with mock.patch.object(self.acs, "download_acs_zcta", return_value=result_paths) as download:
            with redirect_stdout(io.StringIO()):
                code = self.acs.main(
                    ["--output-dir", "unused", "--year", "2021", "--source", "summary-file"]
                )
        self.assertEqual(0, code)
        self.assertEqual("summary-file", download.call_args.kwargs["source"])
        self.assertEqual(2021, download.call_args.kwargs["year"])

    def test_cli_help_requires_no_network(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--year", result.stdout)
        self.assertIn("--source", result.stdout)


if __name__ == "__main__":
    unittest.main()
