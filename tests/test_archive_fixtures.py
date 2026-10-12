"""The shared archive fixtures are what their generator says they are.

The engine and the backend copy tests/fixtures/archives/ byte for byte; a
.tar edited by hand, or a case added without its file, would drift silently.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "archives"


def _generator():
    spec = importlib.util.spec_from_file_location("build_fixtures", FIXTURES / "build_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_committed_archive_is_what_the_generator_builds(tmp_path: Path) -> None:
    generator = _generator()
    generator.HERE = tmp_path
    generator.main()
    for case in json.loads((FIXTURES / "archives.cases.json").read_text())["cases"]:
        assert (FIXTURES / case["file"]).read_bytes() == (tmp_path / case["file"]).read_bytes(), case["file"]
    assert (FIXTURES / "archives.cases.json").read_text() == (tmp_path / "archives.cases.json").read_text()


def test_every_case_has_a_reason_vocabulary_value() -> None:
    cases = json.loads((FIXTURES / "archives.cases.json").read_text())["cases"]
    assert {c["reason"] for c in cases if not c["valid"]} <= {"link", "name", "duplicate", "type"}
    assert all(c["reason"] is None for c in cases if c["valid"])
    assert {p.name for p in FIXTURES.glob("*.tar")} == {c["file"] for c in cases}
