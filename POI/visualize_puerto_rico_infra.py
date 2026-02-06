#!/usr/bin/env python3
"""
Visualize Puerto Rico infrastructure and profile generator attributes.

Outputs:
- POI/puerto_rico_infra_map.html
- Console report for generator:source / generator:method breakdown
"""

import html
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path


REQUIRED_PACKAGES = {
    "folium": "folium",
    "pandas": "pandas",
}


def ensure_dependencies() -> None:
    """Install required packages if missing."""
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            print(f"[DEPS] Installing missing package: {package_name}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            except subprocess.CalledProcessError:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", package_name]
                )


ensure_dependencies()

import folium  # noqa: E402
import pandas as pd  # noqa: E402
from folium.plugins import FastMarkerCluster  # noqa: E402


POI_DIR = Path(__file__).resolve().parent
INPUT_CSV = POI_DIR / "puerto_rico_grid_infra_merged.csv"
OUTPUT_HTML = POI_DIR / "puerto_rico_infra_map.html"

CATEGORY_COLORS = {
    "core_energy": "#d73027",          # red
    "healthcare": "#1f78b4",           # blue
    "water_infrastructure": "#33a02c", # green
    "emergency_shelter": "#ff7f00",    # orange
    "communications": "#6a3d9a",       # violet
    "major_industry": "#8c564b",       # brown
    "tourism_hospitality": "#17becf",  # cyan
    "supply_chain": "#bcbd22",         # olive
}
DEFAULT_COLOR = "#7f7f7f"


def parse_tags(value) -> dict:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def split_multi_values(value: str) -> list:
    if value is None:
        return []
    text = str(value).strip().lower()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[;,|]", text) if p.strip()]
    return parts


def build_marker_rows(df: pd.DataFrame) -> list:
    rows = []
    for row in df.itertuples(index=False):
        try:
            lat = float(row.lat)
            lon = float(row.lon)
        except Exception:
            continue

        tags = parse_tags(row.tags)
        name = row.name if isinstance(row.name, str) and row.name.strip() else "N/A"
        infra_type = row.type if isinstance(row.type, str) and row.type.strip() else "N/A"
        gen_source = tags.get("generator:source", "N/A")

        popup = (
            f"<b>Name:</b> {html.escape(str(name))}<br>"
            f"<b>Type:</b> {html.escape(str(infra_type))}<br>"
            f"<b>generator:source:</b> {html.escape(str(gen_source))}"
        )

        category = str(row.category) if row.category is not None else ""
        color = CATEGORY_COLORS.get(category, DEFAULT_COLOR)

        rows.append([lat, lon, popup, color])

    return rows


def add_category_legend(map_obj: folium.Map) -> None:
    items = "".join(
        [
            (
                "<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<span style='display:inline-block;width:10px;height:10px;background:{color};"
                "margin-right:6px;border:1px solid #333;'></span>"
                f"<span style='font-size:12px;'>{category}</span>"
                "</div>"
            )
            for category, color in CATEGORY_COLORS.items()
        ]
    )
    legend_html = (
        "<div style=\"position: fixed; bottom: 24px; left: 24px; z-index: 9999; "
        "background: white; border: 1px solid #ccc; border-radius: 6px; padding: 10px; "
        "box-shadow: 0 2px 8px rgba(0,0,0,0.25);\">"
        "<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Category Colors</div>"
        f"{items}</div>"
    )
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def create_map(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("Input dataframe is empty, cannot build map.")

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="CartoDB positron",
        control_scale=True,
    )

    marker_rows = build_marker_rows(df)
    callback = """
    function (row) {
      var marker = L.circleMarker([row[0], row[1]], {
        radius: 4,
        color: row[3],
        fillColor: row[3],
        fillOpacity: 0.75,
        weight: 1
      });
      marker.bindPopup(row[2]);
      return marker;
    };
    """
    FastMarkerCluster(marker_rows, callback=callback, name="Infrastructure").add_to(m)
    add_category_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(OUTPUT_HTML))


def print_generator_profile(df: pd.DataFrame) -> None:
    generators = df[df["type"].astype(str).str.lower() == "generator"].copy()

    if generators.empty:
        print("[GENERATOR REPORT] No generator records found.")
        return

    source_counter = Counter()
    method_counter = Counter()
    missing_source = 0
    missing_method = 0

    for row in generators.itertuples(index=False):
        tags = parse_tags(row.tags)

        source_vals = split_multi_values(tags.get("generator:source"))
        method_vals = split_multi_values(tags.get("generator:method"))

        if source_vals:
            for value in source_vals:
                source_counter[value] += 1
        else:
            missing_source += 1

        if method_vals:
            for value in method_vals:
                method_counter[value] += 1
        else:
            missing_method += 1

    print("-" * 78)
    print("[GENERATOR REPORT]")
    print(f"total_generators: {len(generators)}")
    print(f"missing_generator_source: {missing_source}")
    print(f"missing_generator_method: {missing_method}")
    print()
    print("generator:source value counts:")
    if source_counter:
        for key, value in source_counter.most_common():
            print(f"  {key}: {value}")
    else:
        print("  (none)")
    print()
    print("generator:method value counts:")
    if method_counter:
        for key, value in method_counter.most_common():
            print(f"  {key}: {value}")
    else:
        print("  (none)")
    print("-" * 78)


def main() -> int:
    print("[START] Puerto Rico infrastructure visualization + generator profiling")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"id", "category", "type", "lat", "lon", "name", "city", "tags"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    create_map(df)
    print(f"[SAVE] Map written: {OUTPUT_HTML}")

    print_generator_profile(df)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        traceback.print_exc()
        sys.exit(1)
