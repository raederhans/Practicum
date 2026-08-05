from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT_ROOT / "script" / "requirements-stage3.txt"


def test_stage3_spatial_runtime_is_version_pinned():
    lines = {
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    required = {
        "numpy==2.4.6",
        "pandas==3.0.3",
        "geopandas==1.1.3",
        "rasterio==1.5.0",
        "statsmodels==0.14.6",
        "libpysal==4.15.0",
        "esda==2.10.0",
        "spreg==1.9.1",
    }
    assert required <= lines
    assert all("==" in line for line in lines)
