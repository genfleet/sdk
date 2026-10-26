"""Reading the content a workspace supplies to an agent: ``loaded_files``.

A workspace that installs an agent can give it content: a persona, rules,
knowledge, documents. On the platform it is mounted read-only at ``/content``
(``$CONTENT_DIR`` when set). Each file has a mode:

- ``loaded``: meant to go into the agent's context as it is. The platform
  keeps these within a size budget, so an agent can put all of them in its
  prompt.
- ``data``: a file the agent's own code reads by path (declared by the author
  in ``genfleet.toml`` ``[content] data``). Never returned here.

The platform writes the modes next to the files, in ``.genfleet-modes.json``.
With no modes file (a local run, or a folder you made by hand), every text
file under the root counts as ``loaded``; pass ``exclude=`` the same globs as
your ``[content] data`` to leave your data files out.

``loaded_prompt()`` renders the loaded files as one block with a header per
file, ready to go into a system prompt.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

__all__ = [
    "CONTENT_DIR_ENV",
    "DEFAULT_CONTENT_DIR",
    "MODES_FILE",
    "TEXT_SUFFIXES",
    "ContentError",
    "content_glob_problem",
    "content_root",
    "loaded_files",
    "loaded_prompt",
    "matches_content_glob",
]

CONTENT_DIR_ENV = "CONTENT_DIR"
DEFAULT_CONTENT_DIR = "/content"
MODES_FILE = ".genfleet-modes.json"
_MODES_VERSION = 1
_MODES = frozenset({"loaded", "data"})

#: Files read as text. Anything else (images, PDFs, archives) is never
#: returned by ``loaded_files``, whatever its mode.
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".json", ".toml", ".yaml", ".yml", ".csv", ".html", ".htm"})

_MAX_GLOB_CHARS = 512
_GLOB_SPECIALS = frozenset("[]{}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class ContentError(ValueError):
    """The content folder or its modes file is not what the platform writes.

    The message names the file and the problem, never the file's text.
    """


def content_root() -> Path:
    """The content folder: ``$CONTENT_DIR``, else ``/content``."""
    return Path(os.environ.get(CONTENT_DIR_ENV) or DEFAULT_CONTENT_DIR)


# -- globs ----------------------------------------------------------------


def content_glob_problem(glob: object) -> str | None:
    """Why ``glob`` is not a valid content glob, or ``None`` when it is.

    A glob is a path relative to the content root, with ``/`` between
    folders. ``*`` matches any characters within one name, ``?`` one
    character, and a ``**`` segment any number of folders (``flows/**``,
    ``**/*.json``). Brackets and braces are not supported.
    """
    if not isinstance(glob, str) or not glob:
        return "must be a non-empty string"
    if len(glob) > _MAX_GLOB_CHARS:
        return f"is longer than {_MAX_GLOB_CHARS} characters"
    if _CONTROL.search(glob):
        return "has a control character"
    if "\\" in glob:
        return "uses a backslash; separate folders with '/'"
    if glob.startswith("/"):
        return "must be relative to the content root"
    if _GLOB_SPECIALS & set(glob):
        return "uses brackets or braces, which are not supported"
    for segment in glob.split("/"):
        if segment in ("", ".", ".."):
            return "has an empty, '.' or '..' folder"
        if "**" in segment and segment != "**":
            return "uses '**' inside a name; '**' must be a whole folder segment"
    return None


def _glob_regex(glob: str) -> re.Pattern[str]:
    segments = glob.split("/")
    out = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            out.append(r"[^/]+(?:/[^/]+)*" if last else r"(?:[^/]+/)*")
            continue
        out.append("".join(r"[^/]*" if ch == "*" else r"[^/]" if ch == "?" else re.escape(ch) for ch in segment))
        if not last:
            out.append("/")
    return re.compile("".join(out))


def matches_content_glob(path: str, glob: str) -> bool:
    """Whether the content path ``path`` matches ``glob`` (case-sensitive)."""
    problem = content_glob_problem(glob)
    if problem is not None:
        raise ValueError(f"content glob {glob!r} {problem}")
    return _glob_regex(glob).fullmatch(path) is not None


def _excluded(path: str, globs: list[re.Pattern[str]]) -> bool:
    return any(pattern.fullmatch(path) for pattern in globs)


# -- reading --------------------------------------------------------------


def _check_path(path: object) -> str:
    if not isinstance(path, str) or not path:
        raise ContentError(f"{MODES_FILE}: a path is not a non-empty string")
    shown = repr(path[:200])
    if _CONTROL.search(path) or "\\" in path or path.startswith("/"):
        raise ContentError(f"{MODES_FILE}: path {shown} is not a plain relative path")
    if any(segment in ("", ".", "..") for segment in path.split("/")):
        raise ContentError(f"{MODES_FILE}: path {shown} is not a plain relative path")
    return path


def _inside(root: Path, path: str) -> Path:
    """``root/path``, refusing anything that resolves outside ``root`` (a symlink included)."""
    resolved_root = root.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(resolved_root) or target == resolved_root:
        raise ContentError(f"{path!r} is outside the content folder")
    return target


def _read_modes(root: Path) -> dict[str, str] | None:
    modes_path = root / MODES_FILE
    if not modes_path.is_file():
        return None
    try:
        raw = json.loads(modes_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentError(f"{MODES_FILE} can't be read: {type(exc).__name__}") from None
    if not isinstance(raw, dict) or raw.get("version") != _MODES_VERSION or not isinstance(raw.get("files"), dict):
        raise ContentError(f"{MODES_FILE} is not a version {_MODES_VERSION} modes file")
    modes: dict[str, str] = {}
    for path, mode in raw["files"].items():
        _check_path(path)
        if mode not in _MODES:
            raise ContentError(f"{MODES_FILE}: {path!r} has an unknown mode")
        modes[path] = mode
    return modes


def _walk(root: Path) -> list[str]:
    """Every regular file under ``root``, hidden names and symlinks skipped."""
    found = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".") and not (Path(folder) / d).is_symlink()]
        for name in files:
            full = Path(folder) / name
            if name.startswith(".") or full.is_symlink() or not full.is_file():
                continue
            found.append(PurePosixPath(full.relative_to(root)).as_posix())
    return found


def _text(full: Path) -> str | None:
    if full.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        text = full.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text.removeprefix("﻿")


def loaded_files(
    root: str | os.PathLike[str] | None = None,
    *,
    exclude: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """The ``loaded`` text files under the content root, as ``[(path, text)]``.

    Paths are relative, with ``/`` between folders, sorted. Only text files
    (``TEXT_SUFFIXES``, valid UTF-8) are returned; ``data`` files never are.
    With no modes file every text file counts as loaded. ``exclude`` takes
    content globs (as in ``[content] data``) to leave out either way. A
    missing root gives ``[]``.

    Raises ``ContentError`` when the modes file is malformed or names a path
    that escapes the content folder or isn't there.
    """
    base = Path(root) if root is not None else content_root()
    globs = []
    for glob in exclude:
        problem = content_glob_problem(glob)
        if problem is not None:
            raise ValueError(f"exclude: content glob {glob!r} {problem}")
        globs.append(_glob_regex(glob))
    if not base.is_dir():
        return []
    modes = _read_modes(base)
    paths = _walk(base) if modes is None else [path for path, mode in modes.items() if mode == "loaded"]
    result = []
    for path in sorted(paths):
        if _excluded(path, globs):
            continue
        full = _inside(base, path)
        if not full.is_file():
            raise ContentError(f"{path!r} is listed as loaded but is not a file in the content folder")
        text = _text(full)
        if text is not None:
            result.append((path, text))
    return result


def loaded_prompt(
    root: str | os.PathLike[str] | None = None,
    *,
    exclude: Iterable[str] = (),
) -> str:
    """The loaded files as one block, each under a ``--- path ---`` header.

    An empty string when there are none.
    """
    return "\n\n".join(f"--- {path} ---\n{text.rstrip()}" for path, text in loaded_files(root, exclude=exclude))
