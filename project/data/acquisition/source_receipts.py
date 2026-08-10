#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from urllib.request import Request, urlopen
from urllib.parse import urlencode


USER_AGENT = (
    "Practicum-source-acquisition/1.0 "
    "(+https://github.com/raederhans/Practicum)"
)
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
MAX_OVERPASS_RESPONSE_BYTES = 64 * 1024 * 1024
AUTH_ENV_NAMES = (
    "CLOUDSDK_CORE_PROJECT",
    "EARTHENGINE_PROJECT",
    "EARTHENGINE_TOKEN",
    "EE_PROJECT",
    "GCLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
)


class AcquisitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    source_id: str
    filename: str
    url: str
    immutable_identifier: str
    license: str
    scope: str
    expected_bytes: int
    expected_sha256: str | None
    estimated_bytes: int
    quota_or_cost_risk: str
    stop_conditions: tuple[str, ...]
    variant: str | None = None


PUBLIC_ASSETS = (
    AssetSpec(
        asset_id="tiger-2020-zcta520",
        source_id="census_tiger_zcta_county",
        filename="tl_2020_us_zcta520.zip",
        url=(
            "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/"
            "tl_2020_us_zcta520.zip"
        ),
        immutable_identifier="TIGER/Line 2020 ZCTA520 national archive",
        license="U.S. Census Bureau public data; cite source and vintage",
        scope="United States national ZCTA520 geometry, 2020 vintage",
        expected_bytes=527_995_578,
        expected_sha256=(
            "fb91d692a3140a366aa0c188f081f44cc860cccd693859858f997cf14afead9b"
        ),
        estimated_bytes=527_995_578,
        quota_or_cost_risk="No fee; approximately 528 MB local storage and transfer",
        stop_conditions=(
            "HTTP failure",
            "response exceeds expected byte count",
            "size mismatch",
            "checksum mismatch",
            "target already exists",
        ),
    ),
    AssetSpec(
        asset_id="tiger-2020-county",
        source_id="census_tiger_zcta_county",
        filename="tl_2020_us_county.zip",
        url=(
            "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/"
            "tl_2020_us_county.zip"
        ),
        immutable_identifier="TIGER/Line 2020 County national archive",
        license="U.S. Census Bureau public data; cite source and vintage",
        scope="United States national county geometry, 2020 vintage",
        expected_bytes=80_644_766,
        expected_sha256=(
            "a490d33145b8cd308b0b53113d4bb31575b84a2b4cf6ec28fa5855be37559d8d"
        ),
        estimated_bytes=80_644_766,
        quota_or_cost_risk="No fee; approximately 81 MB local storage and transfer",
        stop_conditions=(
            "HTTP failure",
            "response exceeds expected byte count",
            "size mismatch",
            "checksum mismatch",
            "target already exists",
        ),
    ),
    AssetSpec(
        asset_id="worldpop-tur-2020-unconstrained-100m",
        source_id="worldpop",
        filename="tur_ppp_2020.tif",
        url=(
            "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/"
            "TUR/tur_ppp_2020.tif"
        ),
        immutable_identifier=(
            "WorldPop catalog id 6443; DOI 10.5258/SOTON/WP00645; TUR 2020"
        ),
        license=(
            "CC BY 4.0; dataset-specific ODbL obligations may apply to OSM or "
            "building-derived layers"
        ),
        scope="Turkey population count raster, 2020, WGS84, approximately 100m",
        expected_bytes=535_412_088,
        expected_sha256=(
            "1cfa3f38d5daa81aeffb4d66d9b571cc26c7e9585335b25596b37f342d60d49e"
        ),
        estimated_bytes=535_412_088,
        quota_or_cost_risk="No fee; approximately 535 MB local storage and transfer",
        stop_conditions=(
            "HTTP failure",
            "response exceeds expected byte count",
            "size mismatch",
            "checksum mismatch",
            "target already exists",
        ),
        variant=(
            "Unconstrained individual countries 2000-2020; 100m; population "
            "count; not UN-adjusted"
        ),
    ),
    AssetSpec(
        asset_id="worldpop-bhs-2020-unconstrained-100m",
        source_id="worldpop",
        filename="bhs_ppp_2020.tif",
        url=(
            "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/"
            "BHS/bhs_ppp_2020.tif"
        ),
        immutable_identifier=(
            "WorldPop catalog id 6483; DOI 10.5258/SOTON/WP00645; BHS 2020"
        ),
        license=(
            "CC BY 4.0; dataset-specific ODbL obligations may apply to OSM or "
            "building-derived layers"
        ),
        scope="Bahamas population count raster, 2020, WGS84, approximately 100m",
        expected_bytes=13_662_543,
        expected_sha256=(
            "294e4af82f7ed836c7f6651705574555ef58978ee064ce91cc12301f0753a6ab"
        ),
        estimated_bytes=13_662_543,
        quota_or_cost_risk="No fee; approximately 14 MB local storage and transfer",
        stop_conditions=(
            "HTTP failure",
            "response exceeds expected byte count",
            "size mismatch",
            "checksum mismatch",
            "target already exists",
        ),
        variant=(
            "Unconstrained individual countries 2000-2020; 100m; population "
            "count; not UN-adjusted"
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plan(asset: AssetSpec, output_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "planned",
        "network_started": False,
        "asset": asdict(asset),
        "output_filename": asset.filename,
        "receipt_filename": f"{asset.filename}.receipt.json",
        "output_directory": str(output_dir.resolve()),
    }


def acquire_asset(
    asset: AssetSpec,
    *,
    output_dir: Path,
    execute: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    plan = _plan(asset, output_dir)
    if not execute:
        return plan
    if asset.expected_sha256 is None:
        raise AcquisitionError(f"source-checksum-not-pinned:{asset.asset_id}")

    target = output_dir / asset.filename
    partial = output_dir / f"{asset.filename}.partial"
    receipt_path = output_dir / f"{asset.filename}.receipt.json"
    if target.exists():
        raise AcquisitionError(f"target-exists:{target.name}")
    if receipt_path.exists():
        raise AcquisitionError(f"receipt-exists:{receipt_path.name}")
    if partial.exists():
        raise AcquisitionError(f"partial-exists:{partial.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    request = Request(asset.url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=120) as response, partial.open("xb") as handle:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > asset.expected_bytes:
                    raise AcquisitionError(
                        f"response-too-large:{total}>{asset.expected_bytes}"
                    )
                handle.write(chunk)

        observed_bytes = partial.stat().st_size
        if observed_bytes != asset.expected_bytes:
            raise AcquisitionError(
                f"size-mismatch:{observed_bytes}!={asset.expected_bytes}"
            )
        observed_sha256 = _sha256(partial)
        if observed_sha256 != asset.expected_sha256:
            raise AcquisitionError(
                f"checksum-mismatch:{observed_sha256}!={asset.expected_sha256}"
            )

        partial.replace(target)
        receipt = {
            "schema_version": 1,
            "status": "verified",
            "asset_id": asset.asset_id,
            "source_id": asset.source_id,
            "url": asset.url,
            "immutable_identifier": asset.immutable_identifier,
            "variant": asset.variant,
            "license": asset.license,
            "scope": asset.scope,
            "retrieved_at": _utc_now(),
            "filename": asset.filename,
            "bytes": observed_bytes,
            "sha256": observed_sha256,
            "storage": "caller-owned-output-directory",
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def verify_existing_asset(
    asset: AssetSpec,
    *,
    output_dir: Path,
) -> dict[str, object]:
    """Verify a caller-owned prior download without performing network I/O."""
    output_dir = Path(output_dir)
    target = output_dir / asset.filename
    receipt_path = output_dir / f"{asset.filename}.receipt.json"
    if asset.expected_sha256 is None:
        raise AcquisitionError(f"source-checksum-not-pinned:{asset.asset_id}")
    if not target.is_file():
        raise AcquisitionError(f"target-missing:{target.name}")
    if receipt_path.exists():
        raise AcquisitionError(f"receipt-exists:{receipt_path.name}")

    observed_bytes = target.stat().st_size
    if observed_bytes != asset.expected_bytes:
        raise AcquisitionError(
            f"size-mismatch:{observed_bytes}!={asset.expected_bytes}"
        )
    observed_sha256 = _sha256(target)
    if observed_sha256 != asset.expected_sha256:
        raise AcquisitionError(
            f"checksum-mismatch:{observed_sha256}!={asset.expected_sha256}"
        )

    receipt = {
        "schema_version": 1,
        "status": "verified",
        "verification_mode": "existing-file-size-and-sha256",
        "network_started": False,
        "asset_id": asset.asset_id,
        "source_id": asset.source_id,
        "url": asset.url,
        "immutable_identifier": asset.immutable_identifier,
        "variant": asset.variant,
        "license": asset.license,
        "scope": asset.scope,
        "verified_at": _utc_now(),
        "filename": asset.filename,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "storage": "caller-owned-output-directory",
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def build_osm_query(*, bbox: Sequence[float], as_of: str) -> str:
    if len(bbox) != 4:
        raise ValueError("bbox must contain west,south,east,north")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", as_of):
        raise ValueError("as_of must be UTC ISO 8601 without fractional seconds")
    west, south, east, north = bbox
    bbox_ql = f"{south},{west},{north},{east}"
    selectors = (
        'nwr["amenity"="hospital"]',
        'nwr["aeroway"="aerodrome"]',
        'nwr["amenity"="fire_station"]',
        'nwr["amenity"="police"]',
        'nwr["power"="plant"]',
        'nwr["office"="government"]',
        'nwr["amenity"="townhall"]',
        'nwr["amenity"="courthouse"]',
        'nwr["power"="substation"]',
        'nwr["man_made"="water_works"]',
        'nwr["man_made"="wastewater_plant"]',
    )
    body = "\n".join(f"  {selector}({bbox_ql});" for selector in selectors)
    return (
        f'[out:json][timeout:60][date:"{as_of}"];\n'
        f"(\n{body}\n);\n"
        "out center meta;\n"
    )


def _fetch_overpass(
    endpoint: str,
    query: str,
    user_agent: str,
    *,
    max_response_bytes: int = MAX_OVERPASS_RESPONSE_BYTES,
) -> bytes:
    if max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be positive")
    body = urlencode({"data": query}).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    chunks: list[bytes] = []
    observed_bytes = 0
    with urlopen(request, timeout=120) as response:
        while True:
            chunk = response.read(min(1024 * 1024, max_response_bytes - observed_bytes + 1))
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > max_response_bytes:
                raise AcquisitionError(
                    f"response-too-large:{observed_bytes}>{max_response_bytes}"
                )
            chunks.append(chunk)
    return b"".join(chunks)


def acquire_osm_snapshot(
    *,
    events: Mapping[str, Sequence[float]],
    as_of: str,
    output_dir: Path,
    execute: bool = False,
    endpoint: str = OVERPASS_ENDPOINT,
    user_agent: str = USER_AGENT,
    pause_seconds: float = 2.0,
    max_response_bytes: int = MAX_OVERPASS_RESPONSE_BYTES,
    fetcher: Callable[[str, str, str], bytes] | None = None,
) -> dict[str, object]:
    if not events:
        raise AcquisitionError("event-scope-empty")
    if pause_seconds < 0:
        raise ValueError("pause_seconds must be non-negative")
    if max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be positive")
    output_dir = Path(output_dir)
    event_ids = sorted(events)
    plan = {
        "schema_version": 1,
        "status": "planned",
        "network_started": False,
        "source_id": "osm_overpass",
        "snapshot_as_of": as_of,
        "endpoint": endpoint,
        "event_count": len(event_ids),
        "events": event_ids,
        "output_directory": str(output_dir.resolve()),
        "license": "ODbL 1.0",
        "attribution": "© OpenStreetMap contributors",
        "stop_conditions": [
            "HTTP failure",
            "invalid Overpass payload",
            "existing target or partial file",
            "server rate limit",
        ],
    }
    if not execute:
        return plan

    receipt_path = output_dir / "osm_snapshot_receipt.json"
    if receipt_path.exists():
        raise AcquisitionError("receipt-exists:osm_snapshot_receipt.json")
    targets = {event_id: output_dir / f"{event_id}.osm.json" for event_id in event_ids}
    partials = {
        event_id: output_dir / f"{event_id}.osm.json.partial"
        for event_id in event_ids
    }
    for event_id in event_ids:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", event_id):
            raise AcquisitionError(f"invalid-event-id:{event_id}")
        if targets[event_id].exists():
            raise AcquisitionError(f"target-exists:{targets[event_id].name}")
        if partials[event_id].exists():
            raise AcquisitionError(f"partial-exists:{partials[event_id].name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, object]] = []
    committed_targets: list[Path] = []
    try:
        for index, event_id in enumerate(event_ids):
            query = build_osm_query(bbox=events[event_id], as_of=as_of)
            if fetcher is None:
                raw = _fetch_overpass(
                    endpoint,
                    query,
                    user_agent,
                    max_response_bytes=max_response_bytes,
                )
            else:
                raw = fetcher(endpoint, query, user_agent)
                if len(raw) > max_response_bytes:
                    raise AcquisitionError(
                        f"response-too-large:{len(raw)}>{max_response_bytes}"
                    )
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AcquisitionError(
                    f"invalid-overpass-payload:{event_id}:not-json"
                ) from exc
            if not isinstance(payload, dict) or not isinstance(
                payload.get("elements"), list
            ) or not isinstance(payload.get("osm3s"), dict):
                raise AcquisitionError(
                    f"invalid-overpass-payload:{event_id}:missing-elements-or-osm3s"
                )
            partials[event_id].write_bytes(raw)
            receipts.append(
                {
                    "event_id": event_id,
                    "bbox": list(events[event_id]),
                    "filename": targets[event_id].name,
                    "element_count": len(payload["elements"]),
                    "osm_base_timestamp": payload["osm3s"].get(
                        "timestamp_osm_base"
                    ),
                    "bytes": len(raw),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                }
            )
            if index < len(event_ids) - 1 and pause_seconds:
                time.sleep(pause_seconds)

        for event_id in event_ids:
            partials[event_id].replace(targets[event_id])
            committed_targets.append(targets[event_id])
        receipt = {
            "schema_version": 1,
            "status": "verified",
            "source_id": "osm_overpass",
            "snapshot_as_of": as_of,
            "retrieved_at": _utc_now(),
            "endpoint": endpoint,
            "event_count": len(event_ids),
            "license": "ODbL 1.0",
            "license_url": "https://www.openstreetmap.org/copyright",
            "attribution": "© OpenStreetMap contributors",
            "snapshot_method": "Overpass QL attic date setting with raw JSON receipts",
            "events": receipts,
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt
    except Exception:
        for partial in partials.values():
            partial.unlink(missing_ok=True)
        for target in committed_targets:
            target.unlink(missing_ok=True)
        raise


def load_event_scope(path: Path) -> dict[str, list[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("events"), list
    ):
        raise AcquisitionError("invalid-event-scope-manifest")
    events: dict[str, list[float]] = {}
    for item in payload["events"]:
        if not isinstance(item, dict):
            raise AcquisitionError("invalid-event-scope-entry")
        event_id = item.get("event_id")
        bbox = item.get("bbox")
        if not isinstance(event_id, str) or event_id in events:
            raise AcquisitionError(f"duplicate-or-invalid-event-id:{event_id}")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, (int, float)) for value in bbox)
        ):
            raise AcquisitionError(f"invalid-event-bbox:{event_id}")
        events[event_id] = bbox
    if not events:
        raise AcquisitionError("event-scope-empty")
    return events


def earth_engine_auth_preflight(
    *,
    environ: Mapping[str, str] | None = None,
    credential_candidates: Iterable[Path] = (),
    command_lookup: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    env = os.environ if environ is None else environ
    lookup = command_lookup or (lambda name: shutil.which(name) is not None)
    present_env = {
        name: bool(str(env.get(name, "")).strip())
        for name in sorted(AUTH_ENV_NAMES)
        if name in env
    }
    return {
        "schema_version": 1,
        "earthengine_cli_present": bool(lookup("earthengine")),
        "gcloud_cli_present": bool(lookup("gcloud")),
        "credential_file_present": any(
            Path(candidate).is_file() for candidate in credential_candidates
        ),
        "relevant_environment_present": present_env,
        "credential_content_read": False,
    }


def _asset_by_id(asset_id: str) -> AssetSpec:
    for asset in PUBLIC_ASSETS:
        if asset.asset_id == asset_id:
            return asset
    raise AcquisitionError(f"unknown-asset:{asset_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-first public source acquisition and receipt writer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument(
        "--asset",
        required=True,
        choices=[item.asset_id for item in PUBLIC_ASSETS],
    )
    acquire.add_argument("--output-dir", type=Path, required=True)
    acquire.add_argument("--execute", action="store_true")
    osm = subparsers.add_parser("osm-snapshot")
    osm.add_argument("--events-manifest", type=Path, required=True)
    osm.add_argument("--as-of", required=True)
    osm.add_argument("--output-dir", type=Path, required=True)
    osm.add_argument("--endpoint", default=OVERPASS_ENDPOINT)
    osm.add_argument("--pause-seconds", type=float, default=2.0)
    osm.add_argument(
        "--max-response-bytes",
        type=int,
        default=MAX_OVERPASS_RESPONSE_BYTES,
    )
    osm.add_argument("--execute", action="store_true")
    subparsers.add_parser("auth-preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "acquire":
        report = acquire_asset(
            _asset_by_id(args.asset),
            output_dir=args.output_dir,
            execute=args.execute,
        )
    elif args.command == "osm-snapshot":
        report = acquire_osm_snapshot(
            events=load_event_scope(args.events_manifest),
            as_of=args.as_of,
            output_dir=args.output_dir,
            execute=args.execute,
            endpoint=args.endpoint,
            pause_seconds=args.pause_seconds,
            max_response_bytes=args.max_response_bytes,
        )
    else:
        report = earth_engine_auth_preflight()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
