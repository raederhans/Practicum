#!/usr/bin/env python3
"""Download reproducible Census ACS 5-year ZCTA controls and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "project" / "data" / "raw" / "acs"
DEFAULT_YEAR = 2022
DATASET = "Census ACS 5-year ZCTA"
VARIABLES = {
    "NAME": "NAME",
    "total_pop": "B01003_001E",
    "median_income": "B19013_001E",
}
GEOGRAPHY_HEADER = "zip code tabulation area"
CSV_FIELDS = ["ZCTA5CE20", "total_pop", "median_income", "NAME"]
SUMMARY_GEOGRAPHY_BY_YEAR = {
    2022: {
        "level": "zip code tabulation area",
        "geo_id_prefix": "860Z200US",
    }
}


class DownloadError(RuntimeError):
    """The ACS response could not be retrieved."""


class SchemaError(RuntimeError):
    """The ACS response does not satisfy the expected schema."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_api_url(year: int) -> str:
    if year < 2009 or year > 9999:
        raise ValueError("year must identify an ACS 5-year API vintage")
    query = urllib.parse.urlencode(
        {
            "get": ",".join(VARIABLES.values()),
            "for": "zip code tabulation area:*",
        }
    )
    return f"https://api.census.gov/data/{year}/acs/acs5?{query}"


def build_summary_file_url(year: int, table: str) -> str:
    if year < 2009 or year > 9999:
        raise ValueError("year must identify an ACS 5-year summary-file vintage")
    normalized = table.lower()
    if normalized not in {"b01003", "b19013"}:
        raise ValueError(f"unsupported ACS summary table: {table}")
    return (
        "https://www2.census.gov/programs-surveys/acs/summary_file/"
        f"{year}/table-based-SF/data/5YRData/acsdt5y{year}-{normalized}.dat"
    )


def _fetch(url: str, opener: Callable[..., Any]) -> bytes:
    try:
        with opener(url, timeout=60.0) as response:
            status = getattr(response, "status", 200)
            if status is not None and int(status) >= 400:
                raise DownloadError(f"Census API returned HTTP {status}")
            data = response.read()
    except DownloadError:
        raise
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(f"Census API request failed: {exc}") from exc
    if not isinstance(data, bytes) or not data:
        raise DownloadError("Census API returned an empty response")
    return data


def _estimate(value: Any, variable: str) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    try:
        number = int(text)
    except ValueError as exc:
        raise SchemaError(f"{variable} is not an integer estimate: {value!r}") from exc
    return "" if number < 0 else str(number)


def _parse_rows(response_bytes: bytes) -> list[dict[str, str]]:
    try:
        payload = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"Census API response is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
        raise SchemaError("Census API response must be a non-empty array of rows")
    header = payload[0]
    expected = [*VARIABLES.values(), GEOGRAPHY_HEADER]
    if len(header) != len(set(header)) or any(name not in header for name in expected):
        raise SchemaError(f"Census API header must include exactly one of each required field: {expected}")
    positions = {name: header.index(name) for name in expected}
    rows: list[dict[str, str]] = []
    seen_zctas: set[str] = set()
    for line_number, raw in enumerate(payload[1:], start=2):
        if not isinstance(raw, list) or len(raw) != len(header):
            raise SchemaError(f"Census API row {line_number} does not match header width")
        zcta_raw = str(raw[positions[GEOGRAPHY_HEADER]]).strip()
        if not zcta_raw.isdigit() or not 1 <= len(zcta_raw) <= 5:
            raise SchemaError(f"invalid ZCTA at row {line_number}: {zcta_raw!r}")
        zcta = zcta_raw.zfill(5)
        if zcta in seen_zctas:
            raise SchemaError(f"duplicate ZCTA in Census response: {zcta}")
        seen_zctas.add(zcta)
        name = raw[positions["NAME"]]
        if not isinstance(name, str):
            raise SchemaError(f"NAME must be text at row {line_number}")
        rows.append(
            {
                "ZCTA5CE20": zcta,
                "total_pop": _estimate(raw[positions["B01003_001E"]], "B01003_001E"),
                "median_income": _estimate(raw[positions["B19013_001E"]], "B19013_001E"),
                "NAME": name,
            }
        )
    rows.sort(key=lambda row: row["ZCTA5CE20"])
    return rows


def _parse_summary_table(
    response_bytes: bytes,
    *,
    estimate_header: str,
    margin_header: str,
    geo_id_prefix: str,
) -> dict[str, str]:
    try:
        text = response_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"ACS summary file is not valid UTF-8: {exc}") from exc
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="|")
    try:
        header = next(reader)
    except StopIteration as exc:
        raise SchemaError("ACS summary file is empty") from exc
    expected = ["GEO_ID", estimate_header, margin_header]
    if header != expected:
        raise SchemaError(f"ACS summary header must be exactly {expected}, got {header}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(reader, start=2):
        if len(raw) != len(header):
            raise SchemaError(f"ACS summary row {line_number} does not match header width")
        geo_id = raw[0].strip()
        if not geo_id:
            raise SchemaError(f"ACS summary row {line_number} is missing GEO_ID")
        if not geo_id.startswith(geo_id_prefix):
            continue
        zcta = geo_id[len(geo_id_prefix) :]
        if len(zcta) != 5 or not zcta.isdigit():
            raise SchemaError(f"invalid ZCTA GEO_ID at row {line_number}: {geo_id!r}")
        if zcta in values:
            raise SchemaError(f"duplicate ZCTA in ACS summary file: {zcta}")
        values[zcta] = _estimate(raw[1], estimate_header)
    if not values:
        raise SchemaError("ACS summary file contains no ZCTA rows")
    return values


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _write_outputs(output_dir: Path, csv_path: Path, csv_data: bytes, manifest_path: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="acs-zcta-stage-", dir=output_dir) as temporary:
        stage = Path(temporary)
        staged_csv = stage / csv_path.name
        staged_manifest = stage / manifest_path.name
        staged_csv.write_bytes(csv_data)
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        os.replace(staged_csv, csv_path)
        os.replace(staged_manifest, manifest_path)


def download_acs_zcta(
    output_dir: Path,
    *,
    year: int = DEFAULT_YEAR,
    source: str = "api",
    opener: Callable[..., Any] = urllib.request.urlopen,
    retrieved_at: Callable[[], str] = utc_now,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if source == "api":
        api_url = build_api_url(year)
        response_bytes = _fetch(api_url, opener)
        rows = _parse_rows(response_bytes)
        source_manifest: dict[str, Any] = {
            "source": "api",
            "api_url": api_url,
            "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        }
    elif source == "summary-file":
        geography = SUMMARY_GEOGRAPHY_BY_YEAR.get(year)
        if geography is None:
            supported = ", ".join(str(value) for value in sorted(SUMMARY_GEOGRAPHY_BY_YEAR))
            raise ValueError(
                f"summary-file geography mapping is not verified for {year}; supported years: {supported}"
            )
        population_url = build_summary_file_url(year, "b01003")
        income_url = build_summary_file_url(year, "b19013")
        population_bytes = _fetch(population_url, opener)
        income_bytes = _fetch(income_url, opener)
        population = _parse_summary_table(
            population_bytes,
            estimate_header="B01003_E001",
            margin_header="B01003_M001",
            geo_id_prefix=geography["geo_id_prefix"],
        )
        income = _parse_summary_table(
            income_bytes,
            estimate_header="B19013_E001",
            margin_header="B19013_M001",
            geo_id_prefix=geography["geo_id_prefix"],
        )
        if population.keys() != income.keys():
            missing_income = sorted(population.keys() - income.keys())
            missing_population = sorted(income.keys() - population.keys())
            raise SchemaError(
                "ACS summary ZCTA sets do not match; "
                f"missing_income={missing_income}, missing_population={missing_population}"
            )
        rows = [
            {
                "ZCTA5CE20": zcta,
                "total_pop": population[zcta],
                "median_income": income[zcta],
                "NAME": f"ZCTA5 {zcta}",
            }
            for zcta in sorted(population)
        ]
        source_manifest = {
            "source": "summary-file",
            "responses": [
                {
                    "variable": "total_pop",
                    "field": "B01003_E001",
                    "url": population_url,
                    "sha256": hashlib.sha256(population_bytes).hexdigest(),
                },
                {
                    "variable": "median_income",
                    "field": "B19013_E001",
                    "url": income_url,
                    "sha256": hashlib.sha256(income_bytes).hexdigest(),
                },
            ],
            "derived_fields": {
                "NAME": (
                    "Derived as 'ZCTA5 {ZCTA5CE20}' from GEO_ID prefix "
                    f"{geography['geo_id_prefix']}."
                )
            },
            "geography": geography,
            "citation": (
                f"U.S. Census Bureau, {year} American Community Survey 5-year "
                "table-based summary files, tables B01003 and B19013."
            ),
        }
    else:
        raise ValueError(f"unsupported source: {source}")
    csv_data = _csv_bytes(rows)
    stem = f"acs_zcta_{year}"
    csv_path = output_dir / f"{stem}.csv"
    manifest_path = output_dir / f"{stem}.manifest.json"
    manifest = {
        "dataset": DATASET,
        "year": year,
        "variables": VARIABLES,
        "retrieved_at": retrieved_at(),
        "output_sha256": hashlib.sha256(csv_data).hexdigest(),
        "rows": len(rows),
        "license": "U.S. Census Bureau public data; retain source and vintage attribution.",
        "citation": f"U.S. Census Bureau, {year} American Community Survey 5-year estimates, ZCTA API.",
        **source_manifest,
    }
    _write_outputs(output_dir, csv_path, csv_data, manifest_path, manifest)
    return {"csv_path": csv_path, "manifest_path": manifest_path}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument(
        "--source",
        choices=("api", "summary-file"),
        default="api",
        help="explicit retrieval source; errors never trigger an automatic fallback",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = download_acs_zcta(args.output_dir, year=args.year, source=args.source)
    except (DownloadError, SchemaError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Wrote: {result['csv_path']}")
    print(f"Wrote: {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
