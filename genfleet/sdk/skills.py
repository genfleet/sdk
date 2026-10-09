"""Versioned instructions an agent can read on demand during a turn."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator


_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_GITHUB_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_PINNED_REPO = re.compile(r"(.+)@([0-9a-f]{40})\Z")
_MAX_INSTRUCTIONS = 32_000
_MAX_REFERENCES = 100
_MAX_SKILLS = 100
_MAX_DESCRIPTION = 1024
_MAX_VERSION = 40
_SKILL_ROOTS = ("skills", ".agents/skills", ".claude/skills", ".cursor/skills", "agent/skills")
log = logging.getLogger("genfleet.sdk.skills")


def pinned_revision(source: str) -> str | None:
    """Return the commit suffix of a pinned repository source."""
    match = _PINNED_REPO.fullmatch(source)
    return match.group(2) if match else None


def _block_scalar(lines: list[str], start: int, folded: bool) -> str:
    parts: list[str] = []
    for line in lines[start:]:
        if line and not line[0].isspace():
            break
        parts.append(line.strip())
    return (" " if folded else "\n").join(parts).strip()


def _frontmatter(source: str) -> tuple[dict[str, str], str]:
    source = source.replace("\r\n", "\n").replace("\r", "\n")
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
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ValueError(f"invalid quoted skill field: {key}")
            value = value[1:-1].replace("''", "'") if quote == "'" else value[1:-1].replace('\\"', '"')
        fields[key] = str(value)
    return fields, body.strip()


def _reference_files(directory: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    reference_root = directory / "references"
    if not reference_root.is_dir():
        return references
    if not reference_root.resolve().is_relative_to(directory.resolve()):
        raise ValueError("skill references escape the skill directory")
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
    revision = pinned_revision(source)
    if revision:
        source = source[:-(len(revision) + 1)]
    if Path(source).exists():
        raise ValueError("local skill repositories must be passed as Path")
    if source.startswith(("./", "../", "/")):
        raise ValueError("local skill repositories must be passed as Path")
    if _GITHUB_REPO.fullmatch(source) and source not in _SKILL_ROOTS:
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
        if directory.is_dir() and directory.resolve().is_relative_to(root.resolve()) and (directory / "SKILL.md").is_file()
    )
    for location in _SKILL_ROOTS:
        directory = root / location
        if directory.is_dir() and directory.resolve().is_relative_to(root.resolve()):
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
    audiences: frozenset[str] = Field(default_factory=lambda: frozenset({"operator"}))

    @field_validator("audiences")
    @classmethod
    def valid_audiences(cls, value: frozenset[str]) -> frozenset[str]:
        if not value or not value <= {"operator", "customer"}:
            raise ValueError("skill audiences must contain operator and/or customer")
        return value

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

    @field_validator("description")
    @classmethod
    def bounded_description(cls, value: str) -> str:
        if len(value) > _MAX_DESCRIPTION:
            raise ValueError("skill description is too long")
        return value

    @field_validator("version")
    @classmethod
    def bounded_version(cls, value: str) -> str:
        if len(value) > _MAX_VERSION:
            raise ValueError("skill version is too long")
        return value

    @classmethod
    def from_file(cls, path: str | Path) -> Skill:
        """Snapshot one SKILL.md and its text references."""
        file = Path(path)
        if file.stat().st_size > _MAX_INSTRUCTIONS:
            raise ValueError("skill file is too large")
        fields, instructions = _frontmatter(file.read_text(encoding="utf-8"))
        return cls(**fields, instructions=instructions, references=_reference_files(file.parent))


@contextmanager
def _checkout(source: str | Path) -> Iterator[tuple[Path, str]]:
    url = _repository_url(source)
    revision_pin = pinned_revision(source) if isinstance(source, str) else None
    with tempfile.TemporaryDirectory(prefix="genfleet-skills-") as scratch:
        checkout = Path(scratch) / "repo"
        try:
            git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            if revision_pin:
                subprocess.run(["git", "init", "--quiet", str(checkout)], capture_output=True, check=True, timeout=10)
                subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", url], capture_output=True, check=True, timeout=10)
                subprocess.run(
                    ["git", "-C", str(checkout), "fetch", "--quiet", "--depth", "1", "origin", revision_pin],
                    capture_output=True, check=True, timeout=120, env=git_env,
                )
                subprocess.run(
                    ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
                    capture_output=True, check=True, timeout=30, env=git_env,
                )
            else:
                subprocess.run(
                    ["git", "clone", "--quiet", "--depth", "1", url, str(checkout)],
                    capture_output=True, check=True, timeout=120, env=git_env,
                )
            revision = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                capture_output=True, check=True, text=True, timeout=10,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError("could not fetch skill repository with git") from exc
        yield checkout, revision


def _catalog(checkout: Path, revision: str) -> dict[str, tuple[Path, Skill, SkillOption]]:
    paths = _skill_paths(checkout)
    if not paths:
        raise ValueError("skill repository has no discoverable SKILL.md")
    if len(paths) > _MAX_SKILLS:
        raise ValueError("skill repository has too many skills")
    catalog: dict[str, tuple[Path, Skill, SkillOption]] = {}
    for path in paths:
        try:
            if not path.resolve().is_relative_to(checkout.resolve()):
                raise ValueError("skill file escapes the repository")
            if path.stat().st_size > _MAX_INSTRUCTIONS:
                raise ValueError("skill file is too large")
            fields, instructions = _frontmatter(path.read_text(encoding="utf-8"))
            preview = Skill(**fields, instructions=instructions)
        except (ValueError, UnicodeError) as exc:
            log.warning("skipping invalid skill %s: %s", path.relative_to(checkout), exc)
            continue
        if preview.name in catalog:
            raise ValueError(f"skill repository contains duplicate name: {preview.name}")
        catalog[preview.name] = (
            path, preview, SkillOption(preview.name, preview.description, preview.version, revision)
        )
    if not catalog:
        raise ValueError("skill repository has no valid discoverable SKILL.md")
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
            preview.model_copy(update={"references": _reference_files(path.parent), "source_revision": revision})
            for name, (path, preview, _) in catalog.items() if name in selected
        ]


def discover_skills(source: str | Path) -> list[SkillOption]:
    """List the skills available in a repository for selection."""
    with _checkout(source) as (checkout, revision):
        return [option for _, _, option in _catalog(checkout, revision).values()]


__all__ = ["Skill", "SkillOption", "discover_skills", "load_skills"]
