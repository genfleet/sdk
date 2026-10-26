"""``genfleet.sdk.content``: which files an agent may put in its context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genfleet.sdk.content import (
    MODES_FILE,
    ContentError,
    content_glob_problem,
    content_root,
    loaded_files,
    loaded_prompt,
    matches_content_glob,
)

CASES = json.loads((Path(__file__).parent / "fixtures" / "content-globs.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES["matches"], ids=lambda c: f"{c['glob']}~{c['path']}")
def test_glob_cases(case):
    assert matches_content_glob(case["path"], case["glob"]) is case["match"]


@pytest.mark.parametrize("glob", CASES["invalid"])
def test_invalid_globs(glob):
    assert content_glob_problem(glob) is not None
    with pytest.raises(ValueError):
        matches_content_glob("a", glob)


def _tree(root: Path, files: dict[str, bytes | str]) -> Path:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, str):
            path.write_text(body, encoding="utf-8")
        else:
            path.write_bytes(body)
    return root


def _modes(root: Path, files: dict[str, str]) -> None:
    (root / MODES_FILE).write_text(json.dumps({"version": 1, "files": files}), encoding="utf-8")


def test_without_a_modes_file_every_text_file_is_loaded(tmp_path):
    _tree(
        tmp_path,
        {
            "persona.md": "You are helpful.",
            "rules/tone.txt": "Be brief.",
            "logo.png": b"\x89PNG\r\n\x1a\n\x00",
            "bad.md": b"\xff\xfe not utf-8",
            ".hidden.md": "skip",
            ".git/config.md": "skip",
        },
    )
    assert loaded_files(tmp_path) == [("persona.md", "You are helpful."), ("rules/tone.txt", "Be brief.")]


def test_exclude_leaves_data_globs_out(tmp_path):
    _tree(tmp_path, {"persona.md": "p", "knowledge.json": "{}", "flows/a.flow.json": "{}"})
    assert loaded_files(tmp_path, exclude=["knowledge.json", "flows/**"]) == [("persona.md", "p")]


def test_exclude_refuses_a_bad_glob(tmp_path):
    with pytest.raises(ValueError, match="exclude"):
        loaded_files(tmp_path, exclude=["../x"])


def test_the_modes_file_decides(tmp_path):
    _tree(tmp_path, {"persona.md": "p", "knowledge.json": "{}", "faq.md": "f", "extra.md": "unlisted"})
    _modes(tmp_path, {"persona.md": "loaded", "faq.md": "loaded", "knowledge.json": "data"})
    assert loaded_files(tmp_path) == [("faq.md", "f"), ("persona.md", "p")]


def test_a_loaded_image_is_not_returned(tmp_path):
    _tree(tmp_path, {"logo.png": b"\x89PNG", "persona.md": "p"})
    _modes(tmp_path, {"logo.png": "loaded", "persona.md": "loaded"})
    assert loaded_files(tmp_path) == [("persona.md", "p")]


def test_a_byte_order_mark_is_dropped(tmp_path):
    _tree(tmp_path, {"persona.md": "﻿Hello".encode()})
    assert loaded_files(tmp_path) == [("persona.md", "Hello")]


def test_the_root_defaults_to_content_dir(tmp_path, monkeypatch):
    _tree(tmp_path, {"persona.md": "p"})
    monkeypatch.setenv("CONTENT_DIR", str(tmp_path))
    assert content_root() == tmp_path
    assert loaded_files() == [("persona.md", "p")]


def test_without_content_dir_the_root_is_slash_content(monkeypatch):
    monkeypatch.delenv("CONTENT_DIR", raising=False)
    assert content_root() == Path("/content")


def test_a_missing_root_has_no_files(tmp_path):
    assert loaded_files(tmp_path / "nope") == []
    assert loaded_prompt(tmp_path / "nope") == ""


@pytest.mark.parametrize("path", ["../outside.md", "/etc/passwd", "a/../../x.md", "a\\b.md", "a//b.md", "./a.md", ""])
def test_a_modes_path_that_escapes_is_refused(tmp_path, path):
    _modes(tmp_path, {path: "loaded"})
    with pytest.raises(ContentError):
        loaded_files(tmp_path)


def test_a_symlink_out_of_the_root_is_refused(tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    (root / "persona.md").symlink_to(tmp_path / "secret.md")
    _modes(root, {"persona.md": "loaded"})
    with pytest.raises(ContentError, match="outside"):
        loaded_files(root)


def test_a_symlink_is_skipped_without_a_modes_file(tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    (root / "persona.md").symlink_to(tmp_path / "secret.md")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "x.md").write_text("x", encoding="utf-8")
    (root / "linked").symlink_to(tmp_path / "dir")
    assert loaded_files(root) == []


def test_a_listed_file_that_is_missing_is_refused(tmp_path):
    _modes(tmp_path, {"persona.md": "loaded"})
    with pytest.raises(ContentError, match="not a file"):
        loaded_files(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        json.dumps([]),
        json.dumps({"version": 2, "files": {}}),
        json.dumps({"version": 1, "files": []}),
        json.dumps({"version": 1, "files": {"a.md": "retrieved-ish"}}),
    ],
)
def test_a_malformed_modes_file_is_refused(tmp_path, body):
    (tmp_path / MODES_FILE).write_text(body, encoding="utf-8")
    with pytest.raises(ContentError):
        loaded_files(tmp_path)


def test_an_error_never_quotes_the_file(tmp_path):
    _tree(tmp_path, {"persona.md": "TOP SECRET"})
    (tmp_path / MODES_FILE).write_text("TOP SECRET {", encoding="utf-8")
    with pytest.raises(ContentError) as raised:
        loaded_files(tmp_path)
    assert "TOP SECRET" not in str(raised.value)


def test_loaded_prompt_has_a_header_per_file(tmp_path):
    _tree(tmp_path, {"persona.md": "You are Sam.\n", "rules/tone.md": "Be brief."})
    assert loaded_prompt(tmp_path) == "--- persona.md ---\nYou are Sam.\n\n--- rules/tone.md ---\nBe brief."
