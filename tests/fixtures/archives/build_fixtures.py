"""Build the shared agent/tool archive fixtures and their cases file.

    python tests/fixtures/archives/build_fixtures.py

One rule on every side that reads an artifact (engine delivery, backend
submit, CLI build), decided 2026-10-06 after engine#175:

* member types are an allow-list: regular file or directory only;
* a name with a leading ``/``, any ``..`` segment or a backslash is refused;
  only empty and ``.`` segments are folded (``./pkg//x`` is ``pkg/x``), and a
  name that is only the root (``./``) is skipped when it is a directory and
  refused otherwise;
* two members with one name after folding are refused, directories included.

Deterministic (mtime 0, uid/gid 0, fixed order), so re-running it rewrites
byte-identical files and a diff means the rule's fixtures changed.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST_A = b"[agent]\nname = 'a'\nentry = 'agent:build_agent'\n"
MANIFEST_B = b"[agent]\nname = 'b'\nentry = 'evil:build_agent'\n"


def _info(name: str, kind: bytes = tarfile.REGTYPE, size: int = 0, link: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.size, info.linkname = kind, size, link
    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
    info.mode = 0o755 if kind == tarfile.DIRTYPE else 0o644
    return info


def _file(name: str, body: bytes = MANIFEST_A) -> tuple:
    return (_info(name, size=len(body)), body)


def _dir(name: str) -> tuple:
    return (_info(name, tarfile.DIRTYPE), None)


def _special(name: str, kind: bytes, link: str = "") -> tuple:
    return (_info(name, kind, link=link), None)


def _pax(path: str, body: bytes = MANIFEST_A) -> tuple:
    info = _info("placeholder", size=len(body))
    info.pax_headers = {"path": path}
    return (info, body)


CASES: list[tuple[str, int, list[tuple], bool, str | None, str]] = [
    # (file, format, members, valid, reason, note)
    ("plain.tar", tarfile.GNU_FORMAT,
     [_dir("pkg"), _file("pkg/genfleet.toml"), _file("pkg/agent.py", b"def build_agent():\n    return None\n")],
     True, None, "an ordinary agent archive"),
    ("dot-root.tar", tarfile.GNU_FORMAT,
     [_dir("./"), _file("./pkg/genfleet.toml")],
     True, None, "`tar -C dir .` style: the `./` root entry is skipped, `./pkg/x` is `pkg/x`"),
    ("gnu-longname.tar", tarfile.GNU_FORMAT,
     [_file("pkg/" + "a" * 120 + ".py", b"x = 1\n")],
     True, None, "a GNU long name on a regular file (the CLI writes GNU format)"),
    ("pax-longname.tar", tarfile.PAX_FORMAT,
     [_pax("pkg/" + "b" * 120 + ".py", b"x = 1\n")],
     True, None, "a long name from a pax `path` header (tarfile's default format)"),
    ("dot-root-file.tar", tarfile.GNU_FORMAT,
     [_file("./", b"x")],
     False, "name", "only a directory may be the root entry; a file named `./` is refused"),
    ("symlink-inside.tar", tarfile.GNU_FORMAT,
     [_dir("docs"), _special("pkg", tarfile.SYMTYPE, "docs"),
      _file("pkg/genfleet.toml", MANIFEST_A), _file("docs/genfleet.toml", MANIFEST_B)],
     False, "link", "the engine#175 alias: review reads A, extraction leaves B"),
    ("symlink-outside.tar", tarfile.GNU_FORMAT,
     [_special("pkg", tarfile.SYMTYPE, "../outside"), _file("pkg/x")],
     False, "link", "a link out of the tree"),
    ("hardlink.tar", tarfile.GNU_FORMAT,
     [_file("pkg/genfleet.toml"), _special("pkg/copy.toml", tarfile.LNKTYPE, "pkg/genfleet.toml")],
     False, "link", "an in-tree hardlink"),
    ("gnu-longlink-symlink.tar", tarfile.GNU_FORMAT,
     [_special("pkg/alias", tarfile.SYMTYPE, "b" * 120)],
     False, "link", "a GNU long link on a symlink is still a link"),
    ("absolute-name.tar", tarfile.GNU_FORMAT, [_file("/genfleet.toml")],
     False, "name", "a leading slash is refused, never re-rooted"),
    ("dotdot-inside.tar", tarfile.GNU_FORMAT, [_file("pkg/../genfleet.toml")],
     False, "name", "`..` is refused even when it stays inside"),
    ("dotdot-escape.tar", tarfile.GNU_FORMAT, [_file("../genfleet.toml")],
     False, "name", "`..` out of the tree"),
    ("backslash.tar", tarfile.GNU_FORMAT, [_file("pkg\\genfleet.toml")],
     False, "name", "a backslash is a separator to some readers"),
    ("pax-dotdot.tar", tarfile.PAX_FORMAT, [_pax("../x")],
     False, "name", "the real name in a pax `path` header is checked like any name"),
    ("pax-absolute.tar", tarfile.PAX_FORMAT, [_pax("/x")],
     False, "name", "a pax `path` header with a leading slash"),
    ("duplicate-file.tar", tarfile.GNU_FORMAT,
     [_file("pkg/genfleet.toml", MANIFEST_A), _file("pkg/genfleet.toml", MANIFEST_B)],
     False, "duplicate", "last-one-wins without any link"),
    ("duplicate-dot-spelling.tar", tarfile.GNU_FORMAT,
     [_file("./pkg/genfleet.toml", MANIFEST_A), _file("pkg/genfleet.toml", MANIFEST_B)],
     False, "duplicate", "`./pkg/x` and `pkg/x` are one name"),
    ("duplicate-empty-segment.tar", tarfile.GNU_FORMAT,
     [_file("pkg//genfleet.toml", MANIFEST_A), _file("pkg/genfleet.toml", MANIFEST_B)],
     False, "duplicate", "`pkg//x` and `pkg/x` are one name"),
    ("duplicate-dir.tar", tarfile.GNU_FORMAT,
     [_dir("pkg"), _dir("pkg/"), _file("pkg/genfleet.toml")],
     False, "duplicate", "a repeated directory entry is a duplicate too"),
    ("pax-duplicate.tar", tarfile.PAX_FORMAT,
     [_file("pkg/genfleet.toml", MANIFEST_A), _pax("pkg/genfleet.toml", MANIFEST_B)],
     False, "duplicate", "a pax `path` header naming an earlier member"),
    ("fifo.tar", tarfile.GNU_FORMAT, [_special("pkg/pipe", tarfile.FIFOTYPE)],
     False, "type", "FIFO"),
    ("chrdev.tar", tarfile.GNU_FORMAT, [_special("pkg/tty", tarfile.CHRTYPE)],
     False, "type", "character device"),
    ("blkdev.tar", tarfile.GNU_FORMAT, [_special("pkg/disk", tarfile.BLKTYPE)],
     False, "type", "block device"),
    ("contiguous.tar", tarfile.GNU_FORMAT, [_special("pkg/contig", tarfile.CONTTYPE)],
     False, "type", "contiguous file: `TarInfo.isreg()` accepts it, the allow-list must not"),
    ("sparse.tar", tarfile.GNU_FORMAT, [_special("pkg/sparse", tarfile.GNUTYPE_SPARSE)],
     False, "type", "GNU sparse"),
]


def _build(fmt: int, members: list[tuple]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tar:
        for info, body in members:
            tar.addfile(info, io.BytesIO(body) if body is not None else None)
    return buf.getvalue()


def main() -> None:
    for file, fmt, members, _, _, _ in CASES:
        (HERE / file).write_bytes(_build(fmt, members))
    cases = {
        "contract": "agent/tool archive members, v1 (engine#175)",
        "keep_in_sync": (
            "Source of truth: genfleet/sdk tests/fixtures/archives/ (this file, the .tar "
            "files and build_fixtures.py). Copied byte-identical into the engine "
            "(delivery tests) and the backend (submit-time archive parse)."
        ),
        "notes": [
            "Every reader refuses an archive whose `valid` is false, for the stated `reason`: "
            "link | name | duplicate | type.",
            "Member types are an allow-list: regular file (REGTYPE/AREGTYPE) or directory.",
            "Names: refuse a leading `/`, any `..` segment and a backslash; fold only empty and "
            "`.` segments; a name that is only the root (`./`) is skipped if it is a "
            "directory and refused otherwise.",
            "Duplicates are counted after folding, directories included.",
        ],
        "cases": [
            {"file": file, "valid": valid, "reason": reason, "note": note}
            for file, _, _, valid, reason, note in CASES
        ],
    }
    (HERE / "archives.cases.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
