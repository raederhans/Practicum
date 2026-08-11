import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGES_WORKFLOW = ROOT / ".github" / "workflows" / "deploy-dashboard.yml"
ROOT_VERCEL = ROOT / "vercel.json"
ROOT_VERCEL_IGNORE = ROOT / ".vercelignore"


def test_root_github_pages_workflow_deploys_the_public_observatory():
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")

    assert "project/nightlight-public/**" in workflow
    assert "working-directory: project/nightlight-public" in workflow
    assert "cache-dependency-path: project/nightlight-public/package-lock.json" in workflow
    assert "VITE_BASE_PATH: /${{ github.event.repository.name }}/" in workflow
    assert "npm run verify:public -- --require-dist" in workflow
    assert "path: project/nightlight-public/dist" in workflow
    assert "project/nightlight-dashboard" not in workflow


def test_root_vercel_config_builds_the_public_observatory_without_dashboard_data():
    assert ROOT_VERCEL.exists(), "A repository-root Vercel import needs a root deployment contract."
    config = json.loads(ROOT_VERCEL.read_text(encoding="utf-8"))

    assert config["framework"] == "vite"
    assert config["installCommand"] == "npm --prefix project/nightlight-public ci"
    assert config["buildCommand"] == "npm --prefix project/nightlight-public run validate"
    assert config["outputDirectory"] == "project/nightlight-public/dist"
    assert config["headers"][0]["source"] == "/(.*)"
    header_names = {header["key"] for header in config["headers"][0]["headers"]}
    assert {
        "Content-Security-Policy",
        "Referrer-Policy",
        "X-Content-Type-Options",
        "Permissions-Policy",
    } <= header_names


def test_root_vercel_upload_is_a_strict_public_observatory_allowlist():
    assert ROOT_VERCEL_IGNORE.exists(), (
        "A repository-root Vercel project must restrict the source upload at the same root."
    )
    patterns = [
        line.strip()
        for line in ROOT_VERCEL_IGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert patterns == [
        "/*",
        "!/vercel.json",
        "!/DOCS",
        "/DOCS/*",
        "!/DOCS/archive",
        "/DOCS/archive/*",
        "!/DOCS/archive/p2-p3-solo-evidence-performance-20260810",
        "/DOCS/archive/p2-p3-solo-evidence-performance-20260810/*",
        "!/DOCS/archive/p2-p3-solo-evidence-performance-20260810/p2-evidence.md",
        "!/project",
        "/project/*",
        "!/project/nightlight-public",
        "/project/nightlight-public/*",
        "!/project/nightlight-public/src",
        "!/project/nightlight-public/src/**",
        "!/project/nightlight-public/public",
        "!/project/nightlight-public/public/**",
        "!/project/nightlight-public/scripts",
        "!/project/nightlight-public/scripts/**",
        "!/project/nightlight-public/tests",
        "!/project/nightlight-public/tests/**",
        "!/project/nightlight-public/DOCS",
        "/project/nightlight-public/DOCS/*",
        "!/project/nightlight-public/DOCS/archive",
        "/project/nightlight-public/DOCS/archive/*",
        "!/project/nightlight-public/DOCS/archive/proxy-evidence-phase-20260809",
        "/project/nightlight-public/DOCS/archive/proxy-evidence-phase-20260809/*",
        "!/project/nightlight-public/DOCS/archive/proxy-evidence-phase-20260809/plan.md",
        "!/project/nightlight-public/DOCS/archive/proxy-evidence-phase-20260809/proxy-evaluation-report.md",
        "!/project/nightlight-public/index.html",
        "!/project/nightlight-public/package.json",
        "!/project/nightlight-public/package-lock.json",
        "!/project/nightlight-public/vite.config.js",
        "!/project/nightlight-public/vercel.json",
        "!/project/nightlight-public/LICENSE",
        "!/project/nightlight-public/README.md",
        "!/project/nightlight-public/CREDITS.md",
        "!/project/nightlight-public/DATA_POLICY.md",
        "!/project/nightlight-public/SECURITY.md",
        "!/project/nightlight-public/THIRD_PARTY_NOTICES.md",
        "!/project/nightlight-public/USER_STUDY_PROTOCOL.md",
    ]
