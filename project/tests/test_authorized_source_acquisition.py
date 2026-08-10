from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "project" / "data" / "acquisition" / "source_receipts.py"
OSM_SCOPE = ROOT / "project" / "data" / "manifests" / "osm_modeled_event_scope_v1.json"
ACQUISITION_MANIFEST = (
    ROOT / "project" / "data" / "manifests" / "authorized_source_acquisition_v1.json"
)


def _load_module():
    assert MODULE.is_file(), "authorized source acquisition module is missing"
    spec = importlib.util.spec_from_file_location("practicum_source_receipts", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_catalog_locks_public_assets_before_network_io() -> None:
    module = _load_module()

    assets = {item.asset_id: item for item in module.PUBLIC_ASSETS}

    assert set(assets) == {
        "tiger-2020-zcta520",
        "tiger-2020-county",
        "worldpop-tur-2020-unconstrained-100m",
        "worldpop-bhs-2020-unconstrained-100m",
    }
    assert assets["tiger-2020-zcta520"].expected_bytes == 527_995_578
    assert (
        assets["worldpop-tur-2020-unconstrained-100m"].url
        == "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/"
        "TUR/tur_ppp_2020.tif"
    )
    assert assets["worldpop-tur-2020-unconstrained-100m"].variant == (
        "Unconstrained individual countries 2000-2020; 100m; population count; "
        "not UN-adjusted"
    )
    assert assets["worldpop-tur-2020-unconstrained-100m"].expected_sha256 == (
        "1cfa3f38d5daa81aeffb4d66d9b571cc26c7e9585335b25596b37f342d60d49e"
    )
    assert assets["worldpop-bhs-2020-unconstrained-100m"].expected_sha256 == (
        "294e4af82f7ed836c7f6651705574555ef58978ee064ce91cc12301f0753a6ab"
    )
    assert all(item.license and item.scope and item.stop_conditions for item in assets.values())


def test_download_is_dry_run_by_default_and_requires_explicit_output(tmp_path: Path) -> None:
    module = _load_module()
    asset = module.PUBLIC_ASSETS[0]

    plan = module.acquire_asset(asset, output_dir=tmp_path)

    assert plan["status"] == "planned"
    assert plan["network_started"] is False
    assert list(tmp_path.iterdir()) == []


def test_verified_receipt_is_written_only_after_size_and_checksum_match(
    tmp_path: Path,
) -> None:
    module = _load_module()
    payload = b"bounded test payload\n"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    output = tmp_path / "output"
    asset = module.AssetSpec(
        asset_id="test-asset",
        source_id="test",
        filename="result.bin",
        url=source.as_uri(),
        immutable_identifier="test:v1",
        license="test-only",
        scope="test bytes only",
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        estimated_bytes=len(payload),
        quota_or_cost_risk="none",
        stop_conditions=("checksum mismatch",),
    )

    result = module.acquire_asset(asset, output_dir=output, execute=True)

    assert result["status"] == "verified"
    assert (output / "result.bin").read_bytes() == payload
    receipt = json.loads((output / "result.bin.receipt.json").read_text("utf-8"))
    assert receipt["status"] == "verified"
    assert receipt["sha256"] == asset.expected_sha256
    assert receipt["bytes"] == len(payload)
    assert not (output / "result.bin.partial").exists()


def test_checksum_mismatch_fails_closed_without_final_file_or_receipt(
    tmp_path: Path,
) -> None:
    module = _load_module()
    source = tmp_path / "source.bin"
    source.write_bytes(b"unexpected")
    output = tmp_path / "output"
    asset = module.AssetSpec(
        asset_id="bad-asset",
        source_id="test",
        filename="result.bin",
        url=source.as_uri(),
        immutable_identifier="test:v1",
        license="test-only",
        scope="test bytes only",
        expected_bytes=len(b"unexpected"),
        expected_sha256="0" * 64,
        estimated_bytes=len(b"unexpected"),
        quota_or_cost_risk="none",
        stop_conditions=("checksum mismatch",),
    )

    with pytest.raises(module.AcquisitionError, match="checksum-mismatch"):
        module.acquire_asset(asset, output_dir=output, execute=True)

    assert not (output / "result.bin").exists()
    assert not (output / "result.bin.receipt.json").exists()


def test_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "output"
    output.mkdir()
    target = output / "result.bin"
    target.write_bytes(b"owner data")
    asset = module.AssetSpec(
        asset_id="test-asset",
        source_id="test",
        filename="result.bin",
        url=(tmp_path / "absent.bin").as_uri(),
        immutable_identifier="test:v1",
        license="test-only",
        scope="test bytes only",
        expected_bytes=1,
        expected_sha256="0" * 64,
        estimated_bytes=1,
        quota_or_cost_risk="none",
        stop_conditions=("existing target",),
    )

    with pytest.raises(module.AcquisitionError, match="target-exists"):
        module.acquire_asset(asset, output_dir=output, execute=True)

    assert target.read_bytes() == b"owner data"


def test_existing_task_owned_download_can_be_verified_without_network(
    tmp_path: Path,
) -> None:
    module = _load_module()
    payload = b"downloaded before publisher checksum was pinned\n"
    target = tmp_path / "result.bin"
    target.write_bytes(payload)
    before = target.stat().st_mtime_ns
    asset = module.AssetSpec(
        asset_id="test-asset",
        source_id="test",
        filename="result.bin",
        url="https://example.invalid/result.bin",
        immutable_identifier="test:v1",
        license="test-only",
        scope="test bytes only",
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        estimated_bytes=len(payload),
        quota_or_cost_risk="none",
        stop_conditions=("checksum mismatch",),
    )

    receipt = module.verify_existing_asset(asset, output_dir=tmp_path)

    assert receipt["status"] == "verified"
    assert receipt["verification_mode"] == "existing-file-size-and-sha256"
    assert target.read_bytes() == payload
    assert target.stat().st_mtime_ns == before
    assert (tmp_path / "result.bin.receipt.json").is_file()


def test_osm_query_is_pinned_to_an_attic_timestamp() -> None:
    module = _load_module()

    query = module.build_osm_query(
        bbox=(-84.565, 33.60, -84.215, 33.90),
        as_of="2026-08-10T00:00:00Z",
    )

    assert '[date:"2026-08-10T00:00:00Z"]' in query
    assert '[out:json][timeout:60]' in query
    assert 'nwr["amenity"="hospital"]' in query
    assert "33.6,-84.565,33.9,-84.215" in query
    assert query.rstrip().endswith("out center meta;")


def test_auth_preflight_reports_presence_without_returning_secret_values(
    tmp_path: Path,
) -> None:
    module = _load_module()
    credential = tmp_path / "application_default_credentials.json"
    credential.write_text('{"private_key":"must-not-leak"}', encoding="utf-8")

    report = module.earth_engine_auth_preflight(
        environ={
            "GOOGLE_APPLICATION_CREDENTIALS": str(credential),
            "EARTHENGINE_TOKEN": "must-not-leak",
        },
        credential_candidates=[credential],
        command_lookup=lambda name: name == "earthengine",
    )

    rendered = json.dumps(report, sort_keys=True)
    assert report["earthengine_cli_present"] is True
    assert report["credential_file_present"] is True
    assert report["relevant_environment_present"] == {
        "EARTHENGINE_TOKEN": True,
        "GOOGLE_APPLICATION_CREDENTIALS": True,
    }
    assert "must-not-leak" not in rendered
    assert str(credential) not in rendered


def test_auth_preflight_does_not_enumerate_unrelated_environment_values() -> None:
    module = _load_module()

    class PresenceOnlyEnvironment:
        def __contains__(self, key: object) -> bool:
            return key == "EARTHENGINE_PROJECT"

        def get(self, key: str, default: str = "") -> str:
            return "configured" if key == "EARTHENGINE_PROJECT" else default

        def __iter__(self):
            raise AssertionError("credential preflight must not enumerate the environment")

    report = module.earth_engine_auth_preflight(
        environ=PresenceOnlyEnvironment(),
        command_lookup=lambda _name: False,
    )

    assert report["relevant_environment_present"] == {"EARTHENGINE_PROJECT": True}


def test_overpass_fetch_rejects_a_response_above_the_explicit_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int = -1) -> bytes:
            return b"123456789"

    monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: OversizedResponse())

    with pytest.raises(module.AcquisitionError, match="response-too-large"):
        module._fetch_overpass(
            "https://example.invalid/overpass",
            "[out:json];node(0,0,1,1);out;",
            "test-agent",
            max_response_bytes=8,
        )


def test_osm_snapshot_is_dry_run_by_default(tmp_path: Path) -> None:
    module = _load_module()
    events = {"example": [-84.565, 33.60, -84.215, 33.90]}
    calls = []

    result = module.acquire_osm_snapshot(
        events=events,
        as_of="2026-08-10T00:00:00Z",
        output_dir=tmp_path,
        fetcher=lambda *args: calls.append(args),
    )

    assert result["status"] == "planned"
    assert result["network_started"] is False
    assert result["event_count"] == 1
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_osm_snapshot_receipts_raw_responses_and_query_hashes(tmp_path: Path) -> None:
    module = _load_module()
    events = {
        "alpha": [-84.565, 33.60, -84.215, 33.90],
        "beta": [-80.45, 25.55, -80.05, 25.95],
    }

    def fetcher(endpoint: str, query: str, user_agent: str) -> bytes:
        assert endpoint == "https://overpass-api.de/api/interpreter"
        assert '[date:"2026-08-10T00:00:00Z"]' in query
        assert "Practicum" in user_agent
        event_marker = "alpha" if "33.6,-84.565" in query else "beta"
        return json.dumps(
            {
                "version": 0.6,
                "generator": "Overpass API test",
                "osm3s": {"timestamp_osm_base": "2026-08-10T00:01:00Z"},
                "elements": [{"type": "node", "id": 1, "tags": {"name": event_marker}}],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    result = module.acquire_osm_snapshot(
        events=events,
        as_of="2026-08-10T00:00:00Z",
        output_dir=tmp_path,
        execute=True,
        pause_seconds=0,
        fetcher=fetcher,
    )

    assert result["status"] == "verified"
    assert result["event_count"] == 2
    assert {item["event_id"] for item in result["events"]} == {"alpha", "beta"}
    assert all(len(item["raw_sha256"]) == 64 for item in result["events"])
    assert all(len(item["query_sha256"]) == 64 for item in result["events"])
    assert (tmp_path / "alpha.osm.json").is_file()
    assert (tmp_path / "beta.osm.json").is_file()
    receipt = json.loads(
        (tmp_path / "osm_snapshot_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt == result
    assert receipt["license"] == "ODbL 1.0"
    assert receipt["attribution"] == "© OpenStreetMap contributors"


def test_osm_snapshot_fails_closed_on_invalid_payload(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(module.AcquisitionError, match="invalid-overpass-payload"):
        module.acquire_osm_snapshot(
            events={"alpha": [-84.565, 33.60, -84.215, 33.90]},
            as_of="2026-08-10T00:00:00Z",
            output_dir=tmp_path,
            execute=True,
            pause_seconds=0,
            fetcher=lambda *_: b'{"remark":"runtime error"}',
        )

    assert not (tmp_path / "osm_snapshot_receipt.json").exists()
    assert not (tmp_path / "alpha.osm.json").exists()


def test_osm_snapshot_rolls_back_final_files_if_commit_phase_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    original_replace = Path.replace

    def fail_on_second_commit(source: Path, target: Path) -> Path:
        if source.name == "beta.osm.json.partial":
            raise OSError("simulated commit failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_on_second_commit)

    with pytest.raises(OSError, match="simulated commit failure"):
        module.acquire_osm_snapshot(
            events={
                "alpha": [-84.565, 33.60, -84.215, 33.90],
                "beta": [-80.45, 25.55, -80.05, 25.95],
            },
            as_of="2026-08-10T00:00:00Z",
            output_dir=tmp_path,
            execute=True,
            pause_seconds=0,
            fetcher=lambda *_: json.dumps(
                {
                    "osm3s": {"timestamp_osm_base": "2026-08-10T00:01:00Z"},
                    "elements": [],
                }
            ).encode("utf-8"),
        )

    assert list(tmp_path.iterdir()) == []


def test_modeled_osm_scope_covers_both_active_event_registries() -> None:
    module = _load_module()

    events = module.load_event_scope(OSM_SCOPE)

    assert len(events) == 26
    assert events["maria_sanjuan"] == [-66.22, 18.35, -65.95, 18.52]
    assert events["zeta_atlanta"] == [-84.565, 33.6, -84.215, 33.9]
    assert events["atmos_seattle"] == [-122.505, 47.46, -122.155, 47.76]


def test_osm_cli_is_dry_run_without_execute(tmp_path: Path, capsys) -> None:
    module = _load_module()

    result = module.main(
        [
            "osm-snapshot",
            "--events-manifest",
            str(OSM_SCOPE),
            "--as-of",
            "2026-08-10T00:00:00Z",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "planned"
    assert report["event_count"] == 26
    assert list(tmp_path.iterdir()) == []


def test_acquisition_manifest_keeps_all_seven_boundaries_auditable_and_closed() -> None:
    manifest = json.loads(ACQUISITION_MANIFEST.read_text(encoding="utf-8"))
    boundaries = {item["source_id"]: item for item in manifest["boundaries"]}

    assert manifest["base_commit"] == "ca8292040a402eae1d2e461708a4cc912867efcb"
    assert manifest["full_upstream_ready"] is False
    assert set(boundaries) == {
        "nasa_vnp46a2",
        "nlcd",
        "osm_overpass",
        "census_tiger_zcta_county",
        "miami_dade_open_data",
        "worldpop",
        "eagle_i",
    }
    assert boundaries["census_tiger_zcta_county"]["status"] == "verified-local-cache"
    assert len(boundaries["census_tiger_zcta_county"]["assets"]) == 2
    assert boundaries["worldpop"]["status"] == "verified-local-cache"
    assert len(boundaries["worldpop"]["assets"]) == 2
    assert boundaries["osm_overpass"]["blocker_code"] == "http-429-no-complete-receipt"
    assert boundaries["miami_dade_open_data"]["historical_item_id"] == (
        "31cd319f45544648b59f0418aea60091"
    )
    assert boundaries["eagle_i"]["tracked_derivative_count"] == 52
    assert boundaries["nasa_vnp46a2"]["authentication"] == "interactive-required"
    assert boundaries["nlcd"]["authentication"] == "interactive-required"
    assert "token" not in json.dumps(manifest).lower()
