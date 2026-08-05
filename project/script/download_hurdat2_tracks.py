"""Download NOAA NHC Atlantic HURDAT2 and emit a strict normalized track CSV."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "hurdat2_atlantic_normalized.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "raw" / "hurdat2_atlantic_normalized.manifest.json"
OUTPUT_FIELDS = (
    "SEASON",
    "NAME",
    "LAT",
    "LON",
    "USA_R34_NE",
    "USA_R34_SE",
    "USA_R34_SW",
    "USA_R34_NW",
)
RADIUS_FIELDS = OUTPUT_FIELDS[4:]
HEADER_ID_RE = re.compile(r"^AL\d{6}$")
COORD_RE = re.compile(r"^(\d+(?:\.\d+)?)([NSEW])$")
NOAA_CITATION = (
    "NOAA National Hurricane Center (NHC), Atlantic hurricane database "
    "(HURDAT2), https://www.nhc.noaa.gov/data/#hurdat"
)


def _csv_fields(line):
    return [field.strip() for field in next(csv.reader([line]))]


def _parse_header(line, line_number):
    fields = _csv_fields(line)
    if len(fields) != 4 or fields[-1] != "" or not HEADER_ID_RE.fullmatch(fields[0]):
        raise ValueError(f"invalid HURDAT2 header at line {line_number}")
    if not fields[1]:
        raise ValueError(f"invalid HURDAT2 header name at line {line_number}")
    try:
        track_count = int(fields[2])
    except ValueError as exc:
        raise ValueError(f"invalid HURDAT2 header track count at line {line_number}") from exc
    if track_count <= 0:
        raise ValueError(f"invalid HURDAT2 header track count at line {line_number}")
    return fields[0][-4:], fields[1].upper(), track_count


def _parse_coordinate(value, kind, line_number):
    match = COORD_RE.fullmatch(value)
    expected = "NS" if kind == "latitude" else "EW"
    if not match or match.group(2) not in expected:
        raise ValueError(f"invalid {kind} at line {line_number}")
    number = float(match.group(1))
    limit = 90.0 if kind == "latitude" else 180.0
    if number > limit:
        raise ValueError(f"invalid {kind} at line {line_number}")
    if match.group(2) in "SW":
        number = -number
    return number


def _parse_radius(value, line_number):
    try:
        radius = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid radius at line {line_number}") from exc
    if radius == -999:
        return None
    if radius < 0:
        raise ValueError(f"invalid radius at line {line_number}")
    return radius


def _parse_track(line, line_number, season, name):
    fields = _csv_fields(line)
    if len(fields) != 21:
        raise ValueError(f"invalid HURDAT2 record width at line {line_number}")
    if not re.fullmatch(r"\d{8}", fields[0]) or not re.fullmatch(r"\d{4}", fields[1]):
        raise ValueError(f"invalid track date/time at line {line_number}")
    try:
        datetime.strptime(fields[0] + fields[1], "%Y%m%d%H%M")
        max_wind = int(fields[6])
        min_pressure = int(fields[7])
    except ValueError as exc:
        raise ValueError(f"invalid track date/wind/pressure at line {line_number}") from exc
    # HURDAT2 uses -99 (not -999) for missing maximum sustained wind.
    if max_wind < 0 and max_wind != -99:
        raise ValueError(f"invalid track max wind at line {line_number}")
    if min_pressure < 0 and min_pressure != -999:
        raise ValueError(f"invalid track pressure at line {line_number}")

    lat = _parse_coordinate(fields[4], "latitude", line_number)
    lon = _parse_coordinate(fields[5], "longitude", line_number)
    radii = [_parse_radius(value, line_number) for value in fields[8:12]]
    _parse_radius(fields[20], line_number)  # RMW is the official 21st field.
    row = {
        "SEASON": int(season),
        "NAME": name,
        "LAT": lat,
        "LON": lon,
    }
    row.update(dict(zip(RADIUS_FIELDS, radii)))
    return row


def parse_hurdat2(payload):
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("HURDAT2 response is not ASCII") from exc
    lines = [(number, line) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
    rows = []
    storms = 0
    index = 0
    while index < len(lines):
        header_number, header_line = lines[index]
        season, name, expected_count = _parse_header(header_line, header_number)
        index += 1
        remaining = len(lines) - index
        if remaining < expected_count:
            raise ValueError(
                f"HURDAT2 track count mismatch at line {header_number}: "
                f"expected {expected_count}, found {remaining}"
            )
        for _ in range(expected_count):
            line_number, line = lines[index]
            if HEADER_ID_RE.match(_csv_fields(line)[0]):
                raise ValueError(
                    f"HURDAT2 track count mismatch at line {header_number}: "
                    f"expected {expected_count}"
                )
            rows.append(_parse_track(line, line_number, season, name))
            index += 1
        if index < len(lines) and not HEADER_ID_RE.fullmatch(_csv_fields(lines[index][1])[0]):
            raise ValueError(
                f"HURDAT2 track count mismatch at line {header_number}: "
                f"found records beyond declared count {expected_count}"
            )
        storms += 1
    if not rows:
        raise ValueError("HURDAT2 response contains no track records")
    return rows, storms


def _normalized_csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def download_and_normalize(
    *, url, vintage, output_path, manifest_path, retrieved_at=None, timeout=60
):
    request = Request(url, headers={"User-Agent": "Practicum-HURDAT2/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()

    rows, storms = parse_hurdat2(payload)
    csv_bytes = _normalized_csv_bytes(rows)
    provenance = {
        "dataset": "NOAA NHC Atlantic HURDAT2",
        "url": url,
        "vintage": vintage,
        "retrieved_at": retrieved_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "rows": len(rows),
        "storms": storms,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "output_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "citation": NOAA_CITATION,
    }
    manifest_bytes = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(output_path, csv_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    return provenance


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Official NOAA NHC HURDAT2 text URL.")
    parser.add_argument("--vintage", required=True, help="Published coverage and update date.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Normalized CSV path.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Provenance JSON path.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = download_and_normalize(
        url=args.url,
        vintage=args.vintage,
        output_path=args.output,
        manifest_path=args.manifest,
        timeout=args.timeout,
    )
    print(f"Saved {result['rows']} tracks from {result['storms']} storms to {args.output}")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
