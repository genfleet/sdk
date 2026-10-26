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

Paths and globs are compared NFC-normalised and case-sensitive. The modes
file's contract (``version`` 1, ``files`` a ``{path: mode}`` object) is in
``tests/fixtures/content-modes.json``, which the platform shares.

``loaded_prompt()`` renders the loaded files as one block with a header per
file, ready to go into a system prompt.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import NamedTuple

__all__ = [
    "CONTENT_DIR_ENV",
    "DEFAULT_CONTENT_DIR",
    "MODES_FILE",
    "TEXT_SUFFIXES",
    "MAX_LOADED_FILE_BYTES",
    "MAX_LOADED_TOTAL_BYTES",
    "ContentError",
    "LoadedFile",
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

#: Bounds on what ``loaded_files`` reads: far above any plan's loaded budget
#: (32k tokens is well under 1 MiB of text), well below the platform's 64 MB
#: mounted cap.
MAX_LOADED_FILE_BYTES = 4 * 1024 * 1024
MAX_LOADED_TOTAL_BYTES = 8 * 1024 * 1024

log = logging.getLogger(__name__)

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
    return _glob_regex(_nfc(glob)).fullmatch(_nfc(path)) is not None


def _excluded(path: str, globs: list[re.Pattern[str]]) -> bool:
    return any(pattern.fullmatch(path) for pattern in globs)


# -- reading --------------------------------------------------------------


class LoadedFile(NamedTuple):
    """One loaded file: its content path (NFC, ``/``-separated) and its text."""

    path: str
    text: str


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _shown(path: str) -> str:
    """A path as an error or log line shows it: quoted, at most 200 characters."""
    return repr(path[:200])


def _check_path(path: object) -> str:
    """A modes-file path, checked and NFC-normalised."""
    if not isinstance(path, str) or not path:
        raise ContentError(f"{MODES_FILE}: a path is not a non-empty string")
    shown = _shown(path)
    if _CONTROL.search(path) or "\\" in path or path.startswith("/"):
        raise ContentError(f"{MODES_FILE}: path {shown} is not a plain relative path")
    if any(segment in ("", ".", "..") or segment.startswith(".") for segment in path.split("/")):
        raise ContentError(f"{MODES_FILE}: path {shown} is not a plain relative path")
    return _nfc(path)


def _read_modes(root: Path) -> dict[str, str] | None:
    """The modes file's ``loaded`` and ``data`` paths (NFC), or None when there is no modes file.

    Forward compatible: any integer ``version`` of at least 1 is read, as long
    as ``files`` is still a ``{path: mode}`` object; other keys are ignored,
    and a mode this SDK doesn't know is skipped (never loaded).
    """
    modes_path = root / MODES_FILE
    if modes_path.is_symlink():
        raise ContentError(f"{MODES_FILE} is a symlink")
    if not modes_path.exists():
        return None
    try:
        raw = json.loads(modes_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise ContentError(f"{MODES_FILE} can't be read: {type(exc).__name__}") from None
    version = raw.get("version") if isinstance(raw, dict) else None
    if type(version) is not int or version < _MODES_VERSION or not isinstance(raw.get("files"), dict):
        raise ContentError(f"{MODES_FILE} is not a modes file of version {_MODES_VERSION} or later")
    modes: dict[str, str] = {}
    for path, mode in raw["files"].items():
        key = _check_path(path)
        if key in modes:
            raise ContentError(f"{MODES_FILE}: {_shown(key)} is listed twice")
        if isinstance(mode, str) and mode in _MODES:
            modes[key] = mode
        else:
            modes[key] = "unknown"
    return modes


def _walk(root: Path) -> dict[str, Path]:
    """Every regular file under ``root`` by its NFC content path; hidden names and symlinks skipped."""
    found: dict[str, Path] = {}
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".") and not (Path(folder) / d).is_symlink()]
        for name in files:
            full = Path(folder) / name
            if name.startswith(".") or full.is_symlink() or not full.is_file():
                continue
            found[_nfc(PurePosixPath(full.relative_to(root)).as_posix())] = full
    return found


def _inside(root: Path, full: Path, path: str) -> Path:
    """``full``, refusing anything that resolves outside ``root``."""
    resolved_root = root.resolve()
    target = full.resolve()
    if not target.is_relative_to(resolved_root) or target == resolved_root:
        raise ContentError(f"{_shown(path)} is outside the content folder")
    return target


def _text(full: Path, path: str, size: int) -> str | None:
    if PurePosixPath(path).suffix.lower() not in TEXT_SUFFIXES:
        return None
    if size > MAX_LOADED_FILE_BYTES:
        log.warning("content file %s is over %d bytes; it is not loaded", _shown(path), MAX_LOADED_FILE_BYTES)
        return None
    try:
        text = full.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        log.warning("content file %s is not UTF-8 text; it is not loaded", _shown(path))
        return None
    return text.removeprefix("\ufeff")


def _root(root: str | os.PathLike[str] | None) -> Path | None:
    """The root to read, or None when the default ``/content`` is absent (nothing mounted)."""
    explicit = root is not None or bool(os.environ.get(CONTENT_DIR_ENV))
    base = Path(root) if root is not None else content_root()
    if base.is_dir():
        return base
    if explicit:
        raise ContentError(f"the content folder {str(base)!r} is not a directory")
    return None


def loaded_files(
    root: str | os.PathLike[str] | None = None,
    *,
    exclude: Iterable[str] = (),
) -> list[LoadedFile]:
    """The ``loaded`` text files under the content root, as ``[LoadedFile(path, text)]``.

    Paths are relative, NFC-normalised, with ``/`` between folders, sorted.
    Only text files (``TEXT_SUFFIXES``, valid UTF-8; another file is logged
    and left out) are returned; ``data`` files never are. With no modes file
    every text file counts as loaded. ``exclude`` takes content globs to
    leave out; on the platform the modes file already leaves ``data`` files
    out, so it matters for local runs, where you pass your ``[content] data``.

    The root is ``root``, else ``$CONTENT_DIR``, else ``/content``. Only that
    last default may be absent (``[]``); a root you named that isn't a
    folder raises ``ContentError``.

    A loaded file over ``MAX_LOADED_FILE_BYTES`` (4 MiB), or one that would
    take the loaded files past ``MAX_LOADED_TOTAL_BYTES`` (8 MiB) together, is
    left out with a logged warning (sizes are checked before reading): far
    above any plan's loaded budget, well below the platform's 64 MB mounted
    cap, so they bound what an agent reads without failing its start.

    Raises ``ContentError`` when the modes file is malformed, is a symlink,
    or names a path that is hidden, escapes the content folder, or isn't
    there.
    """
    if isinstance(exclude, (str, bytes)):
        raise TypeError("exclude takes a list of globs, not a single string")
    globs = []
    for glob in exclude:
        problem = content_glob_problem(glob)
        if problem is not None:
            raise ValueError(f"exclude: content glob {glob!r} {problem}")
        globs.append(_glob_regex(_nfc(glob)))
    try:
        base = _root(root)
        if base is None:
            return []
        modes = _read_modes(base)
        on_disk = _walk(base)
        if modes is None:
            paths = list(on_disk)
        else:
            paths = [path for path, mode in modes.items() if mode == "loaded"]
            # Only mounted modes must be there: a mode this SDK doesn't know
            # may name a file that isn't mounted (forward compatibility).
            for path, mode in modes.items():
                if mode in _MODES and path not in on_disk:
                    raise ContentError(
                        f"{_shown(path)} is listed in {MODES_FILE} but is not a file in the content folder"
                    )
        result: list[LoadedFile] = []
        total = 0
        for path in sorted(paths):
            if _excluded(path, globs):
                continue
            full = _inside(base, on_disk[path], path)
            size = full.stat().st_size
            text = _text(full, path, size)
            if text is None:
                continue
            if total + size > MAX_LOADED_TOTAL_BYTES:
                log.warning(
                    "content file %s would pass %d loaded bytes; it is not loaded",
                    _shown(path),
                    MAX_LOADED_TOTAL_BYTES,
                )
                continue
            total += size
            result.append(LoadedFile(path, text))
        return result
    except ContentError:
        raise
    except (OSError, RuntimeError, RecursionError, TypeError) as exc:
        raise ContentError(f"the content folder can't be read: {type(exc).__name__}") from None


def loaded_prompt(
    root: str | os.PathLike[str] | None = None,
    *,
    exclude: Iterable[str] = (),
) -> str:
    """The loaded files as one block, each under a ``--- path ---`` header.

    An empty string when there are none.
    """
    return "\n\n".join(f"--- {f.path} ---\n{f.text.rstrip()}" for f in loaded_files(root, exclude=exclude))
