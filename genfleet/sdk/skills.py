"""Versioned instructions an agent can read on demand during a turn."""

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator


_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_GITHUB_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_MAX_INSTRUCTIONS = 32_000
_MAX_REFERENCES = 100
_SKILL_ROOTS = ("skills", ".agents/skills", ".claude/skills", ".cursor/skills", "agent/skills")


def _block_scalar(lines: list[str], start: int, folded: bool) -> str:
    parts: list[str] = []
    for line in lines[start:]:
        if line and not line[0].isspace():
            break
        parts.append(line.strip())
    return (" " if folded else "\n").join(parts).strip()


def _frontmatter(source: str) -> tuple[dict[str, str], str]:
    if not source.startswith("---\n"):
        raise ValueError("skill file needs YAML-style frontmatter")
    header, separator, body = source[4:].partition("\n---\n")
    if not separator:
        raise ValueError("skill file needs a closing frontmatter delimiter")
    lines = header.splitlines()
    fields: dict[str, str] = {}
    for index, line in enumerate(lines):
        key, colon, raw = line.partition(":")
        if not colon or key not in {"name", "description", "version"}:
            continue
        if key in fields:
            raise ValueError(f"duplicate skill field: {key}")
        value = raw.strip()
        if not value or value in {">", ">-", "|", "|-"}:
            value = _block_scalar(lines, index + 1, not value or value.startswith(">"))
        elif value.startswith(("'", '"')):
            value = ast.literal_eval(value)
        fields[key] = str(value)
    return fields, body.strip()


def _reference_files(directory: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    reference_root = directory / "references"
    if not reference_root.is_dir():
        return references
    for path in sorted(reference_root.rglob("*")):
        if path.name == "SKILL.md" or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            continue
        if len(references) >= _MAX_REFERENCES:
            raise ValueError(f"skill has more than {_MAX_REFERENCES} text references")
        if path.stat().st_size > _MAX_INSTRUCTIONS:
            raise ValueError(f"skill reference is too large: {path.relative_to(directory)}")
        references[path.relative_to(directory).as_posix()] = path.read_text(encoding="utf-8")
    return references


def _repository_url(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.resolve().as_uri()
    if _GITHUB_REPO.fullmatch(source):
        return f"https://github.com/{source.removesuffix('.git')}.git"
    parsed = urlsplit(source)
    if parsed.scheme == "https" and parsed.netloc and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment:
        return source
    raise ValueError("skill repository must be owner/repo, an HTTPS Git URL, or a local Path")


def _skill_paths(root: Path) -> list[Path]:
    paths = [root / "SKILL.md"] if (root / "SKILL.md").is_file() else []
    paths.extend(
        directory / "SKILL.md"
        for directory in root.iterdir()
        if directory.is_dir() and (directory / "SKILL.md").is_file()
    )
    for location in _SKILL_ROOTS:
        directory = root / location
        if directory.is_dir():
            paths.extend(
                path for path in directory.rglob("SKILL.md")
                if len(path.relative_to(directory).parts) <= 4
            )
    return sorted(set(paths))


@dataclass(frozen=True)
class SkillOption:
    name: str
    description: str
    version: str
    source_revision: str


class Skill(BaseModel, frozen=True):
    name: str
    description: str
    instructions: str
    version: str = "1"
    references: dict[str, str] = Field(default_factory=dict)
    source_revision: str | None = None

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("skill name must be a lowercase slug (max 64 characters)")
        return value

    @field_validator("description", "instructions", "version")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("skill fields must not be empty")
        return value

    @field_validator("instructions")
    @classmethod
    def bounded_instructions(cls, value: str) -> str:
        if len(value) > _MAX_INSTRUCTIONS:
            raise ValueError(f"skill instructions exceed {_MAX_INSTRUCTIONS} characters")
        return value

    @classmethod
    def from_file(cls, path: str | Path) -> Skill:
        """Snapshot one SKILL.md and its text references."""
        file = Path(path)
        fields, instructions = _frontmatter(file.read_text(encoding="utf-8"))
        return cls(**fields, instructions=instructions, references=_reference_files(file.parent))


@contextmanager
def _checkout(source: str | Path) -> Iterator[tuple[Path, str]]:
    url = _repository_url(source)
    with tempfile.TemporaryDirectory(prefix="genfleet-skills-") as scratch:
        checkout = Path(scratch) / "repo"
        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--depth", "1", url, str(checkout)],
                capture_output=True, check=True, timeout=120,
            )
            revision = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                capture_output=True, check=True, text=True, timeout=10,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError("could not fetch skill repository with git") from exc
        yield checkout, revision


def _catalog(checkout: Path, revision: str) -> dict[str, tuple[Path, SkillOption]]:
    paths = _skill_paths(checkout)
    if not paths:
        raise ValueError("skill repository has no discoverable SKILL.md")
    catalog: dict[str, tuple[Path, SkillOption]] = {}
    for path in paths:
        if not path.resolve().is_relative_to(checkout.resolve()):
            raise ValueError("skill file escapes the repository")
        fields, instructions = _frontmatter(path.read_text(encoding="utf-8"))
        preview = Skill(**fields, instructions=instructions)
        if preview.name in catalog:
            raise ValueError(f"skill repository contains duplicate name: {preview.name}")
        catalog[preview.name] = (
            path, SkillOption(preview.name, preview.description, preview.version, revision)
        )
    return catalog


def load_skills(source: str | Path, *, names: list[str] | None = None) -> list[Skill]:
    """Snapshot selected skills, or every skill when ``names`` is omitted."""
    if names is not None and not names:
        raise ValueError("names must contain at least one skill, or be omitted for all")
    with _checkout(source) as (checkout, revision):
        catalog = _catalog(checkout, revision)
        selected = set(names) if names is not None else set(catalog)
        missing = selected - catalog.keys()
        if missing:
            raise ValueError(f"unknown skills: {', '.join(sorted(missing))}; available: {', '.join(sorted(catalog))}")
        return [
            Skill.from_file(path).model_copy(update={"source_revision": revision})
            for name, (path, _) in catalog.items() if name in selected
        ]


def discover_skills(source: str | Path) -> list[SkillOption]:
    """List the skills available in a repository for selection."""
    with _checkout(source) as (checkout, revision):
        return [option for _, option in _catalog(checkout, revision).values()]


__all__ = ["Skill", "SkillOption", "discover_skills", "load_skills"]
