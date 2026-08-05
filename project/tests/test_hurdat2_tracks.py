import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "script" / "download_hurdat2_tracks.py"
EXTRA_SCRIPT_PATH = PROJECT_ROOT / "script" / "stage3_modelD_extra_regressions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("download_hurdat2_tracks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _track_line(
    *, date, lat, lon, radii, rmw=-999, max_wind=80, min_pressure=950
):
    fields = [
        date,
        "0000",
        "",
        "HU",
        lat,
        lon,
        str(max_wind),
        str(min_pressure),
        *(str(value) for value in radii),
        "-999",
        "-999",
        "-999",
        "-999",
        "-999",
        "-999",
        "-999",
        "-999",
        str(rmw),
    ]
    assert len(fields) == 21
    return ", ".join(fields)


REAL_1851_TRACK = (
    "18510625, 0000,  , HU, 28.0N, 94.8W, 80, -999, "
    "-999, -999, -999, -999, -999, -999, -999, -999, "
    "-999, -999, -999, -999, -999"
)

REAL_1971_MISSING_WIND_TRACK = (
    "19710708, 1200,  , TD, 30.5N, 96.0W, -99, -999, "
    "-999, -999, -999, -999, -999, -999, -999, -999, "
    "-999, -999, -999, -999, -999"
)


def _sample_bytes():
    lines = [
        "AL012020, ARTHUR, 2, ",
        _track_line(date="20200516", lat="28.0N", lon="94.8W", radii=(30, 20, -999, 0)),
        _track_line(date="20200517", lat="12.5S", lon="45.0E", radii=(-999, -999, -999, -999)),
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def test_official_21_field_track_ends_with_rmw_not_empty_column():
    module = _load_module()
    fields = next(csv.reader([REAL_1851_TRACK]))
    assert len(fields) == 21
    assert fields[-1].strip() == "-999"

    row = module._parse_track(REAL_1851_TRACK, 2, "1851", "UNNAMED")
    assert row["SEASON"] == 1851
    assert row["LAT"] == 28.0
    assert row["LON"] == -94.8


def test_official_missing_max_wind_minus_99_is_accepted():
    module = _load_module()
    row = module._parse_track(
        REAL_1971_MISSING_WIND_TRACK, 31589, "1971", "UNNAMED"
    )
    assert row["SEASON"] == 1971
    assert row["LAT"] == 30.5
    assert row["LON"] == -96.0


@pytest.mark.parametrize("sentinel", [-1, -98, -999])
def test_unofficial_negative_max_wind_values_are_rejected(sentinel):
    module = _load_module()
    line = _track_line(
        date="19710708",
        lat="30.5N",
        lon="96.0W",
        radii=(-999, -999, -999, -999),
        max_wind=sentinel,
        min_pressure=-999,
    )
    with pytest.raises(ValueError, match="max wind"):
        module._parse_track(line, 31589, "1971", "UNNAMED")


def test_pressure_only_accepts_nonnegative_or_minus_999():
    module = _load_module()
    accepted = _track_line(
        date="19710708",
        lat="30.5N",
        lon="96.0W",
        radii=(-999, -999, -999, -999),
        max_wind=-99,
        min_pressure=-999,
    )
    module._parse_track(accepted, 31589, "1971", "UNNAMED")

    rejected = _track_line(
        date="19710708",
        lat="30.5N",
        lon="96.0W",
        radii=(-999, -999, -999, -999),
        max_wind=-99,
        min_pressure=-99,
    )
    with pytest.raises(ValueError, match="pressure"):
        module._parse_track(rejected, 31589, "1971", "UNNAMED")


@pytest.mark.parametrize(
    "line",
    [
        ",".join(REAL_1851_TRACK.split(",")[:-1]),
        REAL_1851_TRACK + ",",
    ],
)
def test_track_record_rejects_20_or_22_fields(line):
    module = _load_module()
    with pytest.raises(ValueError, match="record width"):
        module._parse_track(line, 2, "1851", "UNNAMED")


def test_download_normalizes_tracks_and_writes_provenance_atomically(tmp_path, monkeypatch):
    module = _load_module()
    payload = _sample_bytes()
    monkeypatch.setattr(module, "urlopen", lambda request, timeout: _Response(payload))
    output = tmp_path / "tracks.csv"
    manifest = tmp_path / "tracks.manifest.json"

    result = module.download_and_normalize(
        url="https://www.nhc.noaa.gov/data/hurdat/hurdat2-test.txt",
        vintage="Atlantic 1851-2025; updated 2026-02-27",
        output_path=output,
        manifest_path=manifest,
        retrieved_at="2026-08-05T12:00:00Z",
    )

    rows = list(csv.DictReader(io.StringIO(output.read_text(encoding="utf-8"))))
    assert rows[0] == {
        "SEASON": "2020",
        "NAME": "ARTHUR",
        "LAT": "28.0",
        "LON": "-94.8",
        "USA_R34_NE": "30",
        "USA_R34_SE": "20",
        "USA_R34_SW": "",
        "USA_R34_NW": "0",
    }
    assert rows[1]["LAT"] == "-12.5"
    assert rows[1]["LON"] == "45.0"
    assert all(rows[1][key] == "" for key in module.RADIUS_FIELDS)

    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    assert provenance["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance["url"].startswith("https://www.nhc.noaa.gov/")
    assert provenance["vintage"] == "Atlantic 1851-2025; updated 2026-02-27"
    assert provenance["retrieved_at"] == "2026-08-05T12:00:00Z"
    assert provenance["rows"] == 2
    assert provenance["storms"] == 1
    assert "NOAA" in provenance["citation"] and "NHC" in provenance["citation"]
    assert result == provenance
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"XX012020, ARTHUR, 1, \n", "header"),
        (b"AL012020, ARTHUR, 2, \n" + _track_line(date="20200516", lat="28.0N", lon="94.8W", radii=(1, 2, 3, 4)).encode() + b"\n", "track count"),
        (b"AL012020, ARTHUR, 1, \n" + _track_line(date="20200516", lat="28.0N", lon="94.8W", radii=(1, 2, 3, 4)).encode() + b"\n" + _track_line(date="20200517", lat="29.0N", lon="95.0W", radii=(1, 2, 3, 4)).encode() + b"\n", "track count"),
        (b"AL012020, ARTHUR, 1, \n20200516, 0000, HU\n", "record width"),
        (b"AL012020, ARTHUR, 1, \n" + _track_line(date="20200516", lat="91.0N", lon="94.8W", radii=(1, 2, 3, 4)).encode() + b"\n", "latitude"),
        (b"AL012020, ARTHUR, 1, \n" + _track_line(date="20200516", lat="28.0N", lon="94.8W", radii=(1, -1, 3, 4)).encode() + b"\n", "radius"),
        (b"AL012020, ARTHUR, 1, \n" + _track_line(date="20200516", lat="28.0N", lon="94.8W", radii=(1, 2, 3, 4), rmw=-1).encode() + b"\n", "radius"),
    ],
)
def test_malformed_hurdat2_fails_before_output(tmp_path, monkeypatch, payload, expected):
    module = _load_module()
    monkeypatch.setattr(module, "urlopen", lambda request, timeout: _Response(payload))
    output = tmp_path / "tracks.csv"
    manifest = tmp_path / "tracks.manifest.json"

    with pytest.raises(ValueError, match=expected):
        module.download_and_normalize(
            url="https://www.nhc.noaa.gov/data/hurdat/bad.txt",
            vintage="test",
            output_path=output,
            manifest_path=manifest,
        )

    assert not output.exists()
    assert not manifest.exists()


def test_extra_regression_cli_requires_explicit_track_format():
    completed = subprocess.run(
        [sys.executable, str(EXTRA_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--track-file" in completed.stdout
    assert "--track-format {ibtracs,normalized}" in completed.stdout

    source = EXTRA_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "skiprows=[1] if TRACK_FORMAT == 'ibtracs' else None" in source

    ambiguous = subprocess.run(
        [sys.executable, str(EXTRA_SCRIPT_PATH), "--track-file", "tracks.csv"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ambiguous.returncode != 0
    assert "--track-format is required when --track-file is supplied" in ambiguous.stderr
