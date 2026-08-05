from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "project" / "script"
sys.path.insert(0, str(SCRIPT_DIR))
osm = importlib.import_module("stage3_osm_download")


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {"elements": []}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class OsmDownloadContractTests(unittest.TestCase):
    def test_query_raises_after_final_http_failure(self) -> None:
        post = mock.Mock(return_value=FakeResponse(503))

        with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
            osm.query_overpass(
                [-84.5, 33.6, -84.2, 33.9],
                "hospital",
                ['nwr["amenity"="hospital"]'],
                endpoint="https://example.test/interpreter",
                max_attempts=2,
                retry_delay_seconds=0,
                post=post,
                sleeper=lambda _seconds: None,
            )

        self.assertEqual(2, post.call_count)
        for call in post.call_args_list:
            self.assertIn('nwr["amenity"="hospital"]', call.kwargs["data"]["data"])
            self.assertIn("headers", call.kwargs)
            self.assertEqual(
                osm.DEFAULT_USER_AGENT,
                call.kwargs["headers"]["User-Agent"],
            )

    def test_main_rejects_unknown_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown event"):
                osm.main(
                    [
                        "--events",
                        "not_an_event",
                        "--output-dir",
                        tmp,
                        "--pause-seconds",
                        "0",
                    ]
                )

            self.assertFalse((Path(tmp) / "osm_retrieval_manifest.json").exists())

    def test_main_rejects_blank_user_agent_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                osm, "download_pois_for_event", return_value=osm.pd.DataFrame()
            ) as download:
                with self.assertRaisesRegex(ValueError, "user-agent"):
                    try:
                        osm.main(
                            [
                                "--events",
                                "zeta_atlanta",
                                "--output-dir",
                                tmp,
                                "--user-agent",
                                "   ",
                                "--pause-seconds",
                                "0",
                            ]
                        )
                    except SystemExit:
                        self.fail("--user-agent must be a supported CLI option")

            download.assert_not_called()
            self.assertFalse((Path(tmp) / "osm_retrieval_manifest.json").exists())

    def test_zero_results_write_a_headered_csv_that_stage3_can_read(self) -> None:
        event_id = "zeta_atlanta"
        cfg = osm.EVENTS[event_id]

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                osm, "download_pois_for_event", return_value=osm.pd.DataFrame()
            ):
                osm.main(
                    [
                        "--events",
                        event_id,
                        "--output-dir",
                        tmp,
                        "--pause-seconds",
                        "0",
                    ]
                )

            csv_path = Path(tmp) / f"{cfg['drive_root']}_poi.csv"
            frame = osm.pd.read_csv(csv_path)
            self.assertTrue(frame.empty)
            self.assertEqual(osm.POI_COLUMNS, frame.columns.tolist())

            manifest = json.loads(
                (Path(tmp) / "osm_retrieval_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(0, manifest["events"][0]["row_count"])

    def test_success_writes_sorted_deduplicated_csv_and_manifest(self) -> None:
        event_id = "zeta_atlanta"
        cfg = osm.EVENTS[event_id]
        duplicate = {
            "name": "Zeta Hospital",
            "facility_type": "hospital",
            "lat": 33.8,
            "lon": -84.3,
            "osm_id": 20,
            "osm_type": "node",
        }
        results = {
            "hospital": [
                duplicate,
                {
                    "name": "Alpha Hospital",
                    "facility_type": "hospital",
                    "lat": 33.7,
                    "lon": -84.4,
                    "osm_id": 10,
                    "osm_type": "way",
                },
            ],
            "aerodrome": [
                {**duplicate, "name": "Duplicate from another query", "facility_type": "aerodrome"},
                {
                    "name": "Airport",
                    "facility_type": "aerodrome",
                    "lat": 33.9,
                    "lon": -84.2,
                    "osm_id": 5,
                    "osm_type": "node",
                },
            ],
        }

        def fake_query(_bbox, facility_type, _tags, **kwargs):
            self.assertEqual("https://example.test/interpreter", kwargs["endpoint"])
            self.assertEqual(2, kwargs["max_attempts"])
            self.assertEqual(0, kwargs["retry_delay_seconds"])
            self.assertEqual("Practicum-tests/2.0", kwargs["user_agent"])
            return results.get(facility_type, [])

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(osm, "query_overpass", side_effect=fake_query):
                try:
                    osm.main(
                        [
                            "--events",
                            event_id,
                            "--output-dir",
                            tmp,
                            "--endpoint",
                            "https://example.test/interpreter",
                            "--user-agent",
                            "Practicum-tests/2.0",
                            "--max-attempts",
                            "2",
                            "--retry-delay-seconds",
                            "0",
                            "--pause-seconds",
                            "0",
                        ]
                    )
                except SystemExit:
                    self.fail("--user-agent must be a supported CLI option")

            csv_path = Path(tmp) / f"{cfg['drive_root']}_poi.csv"
            csv_bytes = csv_path.read_bytes()
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(3, len(rows))
            self.assertEqual(
                [("node", "5"), ("node", "20"), ("way", "10")],
                [(row["osm_type"], row["osm_id"]) for row in rows],
            )

            manifest = json.loads(
                (Path(tmp) / "osm_retrieval_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("1.0", manifest["schema_version"])
            self.assertRegex(manifest["retrieved_at"], r"Z$")
            self.assertEqual("https://example.test/interpreter", manifest["endpoint"])
            self.assertEqual("Practicum-tests/2.0", manifest["user_agent"])
            self.assertEqual("OpenStreetMap contributors", manifest["attribution"])
            self.assertEqual("ODbL 1.0", manifest["license"])
            self.assertEqual(1, len(manifest["events"]))

            event = manifest["events"][0]
            self.assertEqual(event_id, event["event_id"])
            self.assertEqual(cfg["drive_root"], event["drive_root"])
            self.assertEqual(cfg["bounds"], event["bbox"])
            self.assertEqual(osm.FACILITY_QUERIES, event["query_tags"])
            self.assertEqual(csv_path.name, event["csv_file"])
            self.assertEqual(3, event["row_count"])
            self.assertEqual(hashlib.sha256(csv_bytes).hexdigest(), event["sha256"])


if __name__ == "__main__":
    unittest.main()
