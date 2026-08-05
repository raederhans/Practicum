"""
stage3_osm_download.py
======================
Download critical facility POIs from OpenStreetMap for Stage 3 events.
Uses Overpass API (free, no key needed).

Usage:
    python stage3_osm_download.py                # all events
    python stage3_osm_download.py --events uri_houston zeta_atlanta

Output:
    data/result/stage3/poi_cache/<event_id>_poi.csv
"""

import os
import time
import argparse
import hashlib
import json
from datetime import datetime, timezone

import requests
import pandas as pd

# Import event configs (no ee dependency)
from stage3_events import EVENTS

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_USER_AGENT = (
    "Practicum-reproducibility/1.0 "
    "(+https://github.com/raederhans/Practicum)"
)

# OSM tags → facility types (same as Stage 2)
FACILITY_QUERIES = {
    "hospital": [
        'nwr["amenity"="hospital"]',
    ],
    "aerodrome": [
        'nwr["aeroway"="aerodrome"]',
    ],
    "fire_station": [
        'nwr["amenity"="fire_station"]',
    ],
    "police": [
        'nwr["amenity"="police"]',
    ],
    "power_plant": [
        'nwr["power"="plant"]',
    ],
    "government": [
        'nwr["office"="government"]',
        'nwr["amenity"="townhall"]',
        'nwr["amenity"="courthouse"]',
    ],
    "substation": [
        'nwr["power"="substation"]',
    ],
    "water_works": [
        'nwr["man_made"="water_works"]',
        'nwr["man_made"="wastewater_plant"]',
    ],
}

POI_COLUMNS = ["name", "facility_type", "lat", "lon", "osm_id", "osm_type"]


def query_overpass(
    bbox,
    facility_type,
    tags,
    *,
    endpoint=OVERPASS_URL,
    user_agent=DEFAULT_USER_AGENT,
    max_attempts=5,
    retry_delay_seconds=15,
    post=None,
    sleeper=time.sleep,
):
    """Query Overpass API for a facility type within a bounding box."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    user_agent = user_agent.strip()
    if not user_agent:
        raise ValueError("user-agent must not be blank")

    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    bbox_str = f"{south},{west},{north},{east}"

    union_parts = "".join(f"{tag}({bbox_str});" for tag in tags)
    query = f"""
    [out:json][timeout:60];
    ({union_parts});
    out center;
    """

    request_post = post or requests.post
    data = None
    last_error = None
    for attempt in range(max_attempts):
        resp = None
        try:
            resp = request_post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": user_agent},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                multiplier = 2 if getattr(resp, "status_code", None) == 429 else 1
                wait = retry_delay_seconds * multiplier * (attempt + 1)
                print(f"    Error: {e}, retrying in {wait}s...")
                sleeper(wait)

    if data is None:
        raise RuntimeError(
            f"Overpass query for {facility_type} failed after {max_attempts} attempts"
        ) from last_error

    results = []
    for el in data.get("elements", []):
        tags_data = el.get("tags", {})
        name = tags_data.get("name", "")

        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue

        results.append({
            "name": name if name else f"{facility_type} facility",
            "facility_type": facility_type,
            "lat": lat,
            "lon": lon,
            "osm_id": el["id"],
            "osm_type": el["type"],
        })

    return results


def download_pois_for_event(
    event_id,
    *,
    endpoint=OVERPASS_URL,
    user_agent=DEFAULT_USER_AGENT,
    max_attempts=5,
    retry_delay_seconds=15,
    pause_seconds=3,
):
    """Download all facility POIs for one event."""
    cfg = EVENTS[event_id]
    bbox = cfg["bounds"]
    print(f"\n  {event_id} — {cfg['name']}")
    print(f"    BBox: {bbox}")

    all_pois = []
    for fac_type, tags in FACILITY_QUERIES.items():
        pois = query_overpass(
            bbox,
            fac_type,
            tags,
            endpoint=endpoint,
            user_agent=user_agent,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        print(f"    {fac_type:<15} {len(pois):>3} POIs")
        all_pois.extend(pois)
        time.sleep(pause_seconds)  # be polite to Overpass

    return pd.DataFrame(all_pois, columns=POI_COLUMNS)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download OSM POIs for Stage 3")
    parser.add_argument("--events", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--endpoint", default=OVERPASS_URL)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay-seconds", type=float, default=15)
    parser.add_argument("--pause-seconds", type=float, default=3)
    args = parser.parse_args(argv)

    event_ids = args.events or list(EVENTS.keys())
    unknown_events = [event_id for event_id in event_ids if event_id not in EVENTS]
    if unknown_events:
        raise ValueError(f"Unknown event(s): {', '.join(unknown_events)}")
    args.user_agent = args.user_agent.strip()
    if not args.user_agent:
        raise ValueError("--user-agent must not be blank")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.retry_delay_seconds < 0 or args.pause_seconds < 0:
        raise ValueError("pause and retry delays must be non-negative")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_dir = os.path.join(project_root, "data", "result", "stage3", "poi_cache")
    os.makedirs(out_dir, exist_ok=True)

    print(f"OSM POI Download — {len(event_ids)} events")
    print(f"Output: {out_dir}")

    summary = []
    manifest_events = []
    for eid in event_ids:
        poi_df = download_pois_for_event(
            eid,
            endpoint=args.endpoint,
            user_agent=args.user_agent,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            pause_seconds=args.pause_seconds,
        )

        if poi_df.empty:
            poi_df = pd.DataFrame(columns=POI_COLUMNS)
        else:
            missing_columns = sorted(set(POI_COLUMNS) - set(poi_df.columns))
            if missing_columns:
                raise ValueError(
                    f"OSM result for {eid} is missing columns: "
                    + ", ".join(missing_columns)
                )
            poi_df = poi_df.loc[:, POI_COLUMNS]

        if not poi_df.empty:
            poi_df = (
                poi_df.drop_duplicates(subset=["osm_type", "osm_id"])
                .sort_values(["osm_type", "osm_id"], kind="stable")
                .reset_index(drop=True)
            )

        out_path = os.path.join(out_dir, f"{EVENTS[eid]['drive_root']}_poi.csv")
        poi_df.to_csv(out_path, index=False)
        print(f"    Saved {len(poi_df)} POIs → {out_path}")
        summary.append({"event": eid, "pois": len(poi_df)})
        manifest_events.append(
            {
                "event_id": eid,
                "drive_root": EVENTS[eid]["drive_root"],
                "bbox": EVENTS[eid]["bounds"],
                "query_tags": FACILITY_QUERIES,
                "csv_file": os.path.basename(out_path),
                "row_count": len(poi_df),
                "sha256": _sha256(out_path),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "endpoint": args.endpoint,
        "user_agent": args.user_agent,
        "attribution": "OpenStreetMap contributors",
        "license": "ODbL 1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "events": manifest_events,
    }
    manifest_path = os.path.join(out_dir, "osm_retrieval_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"\n{'='*40}")
    print(f"{'Event':<22} {'POIs':>6}")
    print("-" * 30)
    for s in summary:
        print(f"  {s['event']:<22} {s['pois']:>4}")
    total = sum(s["pois"] for s in summary)
    print(f"  {'TOTAL':<22} {total:>4}")


if __name__ == "__main__":
    main()
