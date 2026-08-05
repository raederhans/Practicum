import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ARTIFACT = ROOT / 'project' / 'nightlight-public' / 'src' / 'content' / 'generalizationArtifact.js'
PUBLIC_STUDY = ROOT / 'project' / 'nightlight-public' / 'src' / 'content' / 'study.js'
CROSS_EVENT_DECISION = ROOT / 'project' / 'modeling' / 'output' / 'cross_event_stop_decision_v3x.json'


def sha256(path: Path) -> str:
    canonical_bytes = path.read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(canonical_bytes).hexdigest()


def test_sha256_canonicalizes_git_text_line_endings(tmp_path):
    lf_source = tmp_path / 'source-lf.txt'
    crlf_source = tmp_path / 'source-crlf.txt'
    lf_source.write_bytes(b'alpha\nbeta\n')
    crlf_source.write_bytes(b'alpha\r\nbeta\r\n')

    assert sha256(lf_source) == sha256(crlf_source)


def test_public_generalization_source_pointers_match_private_and_public_monorepo_sources():
    """The export gate binds public pointers without copying private source into the public site."""
    contents = PUBLIC_ARTIFACT.read_text(encoding='utf-8')
    sources = (
        ('public-study-summary-v1', 'study.js@1', sha256(PUBLIC_STUDY)),
        ('cross-event-stop-decision-v3x', 'v3x-r1', sha256(CROSS_EVENT_DECISION)),
    )

    for source_id, version, source_hash in sources:
        pattern = rf"id: '{re.escape(source_id)}',\s+version: '{re.escape(version)}',\s+sha256: '{source_hash}'"
        assert re.search(pattern, contents), source_id

    assert 'project/modeling' not in contents
    assert 'cross_event_stop_decision_v3x.json' not in contents
