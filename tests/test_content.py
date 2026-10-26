"""``genfleet.sdk.content``: which files an agent may put in its context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genfleet.sdk.content import (
    MAX_LOADED_FILE_BYTES,
    MAX_LOADED_TOTAL_BYTES,
    MODES_FILE,
    ContentError,
    LoadedFile,
    content_glob_problem,
    content_root,
    loaded_files,
    loaded_prompt,
    matches_content_glob,
)

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "content-globs.json").read_text(encoding="utf-8"))
MODES_CASES = json.loads((FIXTURES / "content-modes.json").read_text(encoding="utf-8"))["cases"]


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
    _tree(tmp_path, {"persona.md": "\ufeffHello".encode()})
    assert loaded_files(tmp_path) == [("persona.md", "Hello")]


def test_the_root_defaults_to_content_dir(tmp_path, monkeypatch):
    _tree(tmp_path, {"persona.md": "p"})
    monkeypatch.setenv("CONTENT_DIR", str(tmp_path))
    assert content_root() == tmp_path
    assert loaded_files() == [("persona.md", "p")]


def test_without_content_dir_the_root_is_slash_content(monkeypatch):
    monkeypatch.delenv("CONTENT_DIR", raising=False)
    assert content_root() == Path("/content")


def test_a_named_root_that_is_missing_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ContentError, match="not a directory"):
        loaded_files(tmp_path / "nope")
    monkeypatch.setenv("CONTENT_DIR", str(tmp_path / "nope"))
    with pytest.raises(ContentError, match="not a directory"):
        loaded_files()


def test_the_default_root_may_be_absent(monkeypatch):
    monkeypatch.delenv("CONTENT_DIR", raising=False)
    monkeypatch.setattr("genfleet.sdk.content.DEFAULT_CONTENT_DIR", "/nonexistent-genfleet-content")
    assert loaded_files() == []
    assert loaded_prompt() == ""


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
    # A symlink is never a content file: refused whether it points in or out.
    with pytest.raises(ContentError, match="not a file"):
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


@pytest.mark.parametrize("case", MODES_CASES, ids=lambda c: c["name"])
def test_the_modes_file_contract(tmp_path, case):
    for rel in case["files"]:
        _tree(tmp_path, {rel: "x"})
    body = case["modes"] if isinstance(case["modes"], str) else json.dumps(case["modes"])
    (tmp_path / MODES_FILE).write_text(body, encoding="utf-8")
    if case.get("error"):
        with pytest.raises(ContentError):
            loaded_files(tmp_path)
    else:
        assert [f.path for f in loaded_files(tmp_path)] == case["loaded"]


def test_a_symlinked_modes_file_is_refused(tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    (tmp_path / "evil.json").write_text(json.dumps({"version": 1, "files": {}}), encoding="utf-8")
    (root / MODES_FILE).symlink_to(tmp_path / "evil.json")
    with pytest.raises(ContentError, match="symlink"):
        loaded_files(root)


def test_a_symlink_loop_is_a_content_error(tmp_path):
    (tmp_path / "a.md").symlink_to(tmp_path / "b.md")
    (tmp_path / "b.md").symlink_to(tmp_path / "a.md")
    _modes(tmp_path, {"a.md": "loaded"})
    with pytest.raises(ContentError):
        loaded_files(tmp_path)


def test_a_bare_string_exclude_is_refused(tmp_path):
    with pytest.raises(TypeError, match="list of globs"):
        loaded_files(tmp_path, exclude="knowledge.json")


def test_a_non_utf8_text_file_is_logged_and_left_out(tmp_path, caplog):
    _tree(tmp_path, {"bad.md": b"\xff\xfe"})
    with caplog.at_level("WARNING", logger="genfleet.sdk.content"):
        assert loaded_files(tmp_path) == []
    assert "bad.md" in caplog.text


def test_a_file_over_the_cap_is_logged_and_left_out_unread(tmp_path, caplog):
    big = tmp_path / "big.md"
    with big.open("wb") as fh:
        fh.truncate(MAX_LOADED_FILE_BYTES + 1)  # sparse: never read
    _tree(tmp_path, {"persona.md": "p"})
    with caplog.at_level("WARNING", logger="genfleet.sdk.content"):
        assert [f.path for f in loaded_files(tmp_path)] == ["persona.md"]
    assert "big.md" in caplog.text


def test_reading_stops_at_the_total_cap_so_the_loaded_set_is_a_prefix(tmp_path, caplog):
    n = MAX_LOADED_TOTAL_BYTES // MAX_LOADED_FILE_BYTES
    for i in range(n + 1):
        (tmp_path / f"f{i}.md").write_bytes(b"a" * MAX_LOADED_FILE_BYTES)
    (tmp_path / "z.md").write_text("small, but after the cap", encoding="utf-8")
    with caplog.at_level("WARNING", logger="genfleet.sdk.content"):
        got = loaded_files(tmp_path)
    assert [f.path for f in got] == [f"f{i}.md" for i in range(n)]
    assert f"f{n}.md" in caplog.text


def test_returns_named_tuples(tmp_path):
    _tree(tmp_path, {"persona.md": "p"})
    (got,) = loaded_files(tmp_path)
    assert isinstance(got, LoadedFile) and got.path == "persona.md" and got.text == "p"


def test_nfd_names_on_disk_are_returned_in_nfc(tmp_path):
    _tree(tmp_path, {"cafe\u0301.md": "x"})
    assert [f.path for f in loaded_files(tmp_path, exclude=["nothing"])] == ["caf\u00e9.md"]
    assert loaded_files(tmp_path, exclude=["caf\u00e9.md"]) == []


def test_an_error_never_quotes_the_file(tmp_path):
    _tree(tmp_path, {"persona.md": "TOP SECRET"})
    (tmp_path / MODES_FILE).write_text("TOP SECRET {", encoding="utf-8")
    with pytest.raises(ContentError) as raised:
        loaded_files(tmp_path)
    assert "TOP SECRET" not in str(raised.value)


def test_loaded_prompt_has_a_header_per_file(tmp_path):
    _tree(tmp_path, {"persona.md": "You are Sam.\n", "rules/tone.md": "Be brief."})
    assert loaded_prompt(tmp_path) == "--- persona.md ---\nYou are Sam.\n\n--- rules/tone.md ---\nBe brief."
