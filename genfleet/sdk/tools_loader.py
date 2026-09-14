"""
`load_tools()` — turn an agent's manifest into callables it can use.

An agent names the tools it wants and receives them; it never imports them.
That indirection is the whole design (ADR-0007): the same published agent runs
for two tenants against their own credentials, because what a name resolves to
is decided at run time rather than at `import` time.

Resolution is pluggable, and the two implementations differ only in where they
look:

* **local** (this module's default) — whatever tool providers are installed,
  secrets from the environment. What a developer gets on a laptop with no
  backend. Providers advertise themselves through an entry point group, so this
  SDK names no particular package.
* **mounted** — tool artifacts the engine already fetched, verified and
  mounted read-only next to the agent. What runs inside a sandbox.
* **platform** — config and secrets from the tenant's installation. Supplied
  by the runtime, not from here.

The mounted resolver is here rather than in the engine because it is only a
filesystem lookup: it reads directories and knows nothing about the registry
that filled them. A resolver that reached into tenant installations, or that
called the registry itself, would drag the platform across the open-source
boundary — and, inside a sandbox, would need a credential ADR-0010 refused to
put there.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Protocol, runtime_checkable

from .manifest import (
    AgentManifest,
    ManifestError,
    ToolManifest,
    ToolRef,
    load_agent_manifest,
    load_tool_manifest,
)


# Set by whatever unpacked the agent — the engine, a sandbox, a test harness.
# Exists because an agent's own idea of where it lives is computed when the
# code is written and can be wrong once the artifact is relocated.
AGENT_ROOT_ENV = "GENFLEET_AGENT_ROOT"

logger = logging.getLogger(__name__)


class ToolResolutionError(RuntimeError):
    """A named tool could not be turned into something callable."""


@runtime_checkable
class ToolResolver(Protocol):
    """How a reference becomes a callable. The platform supplies its own."""

    def resolve(self, ref: ToolRef) -> Callable: ...


# Any installed distribution can advertise tools by publishing an entry point
# in this group. Nothing here names a particular package on purpose: this SDK
# is open-source, and hardcoding one would point a public user at a repository
# they may not be able to install.
#
#     [project.entry-points."genfleet.tools"]
#     genfleet_tools = "genfleet_tools:discover"
#
# The named callable returns {slug: object with .load() -> Callable}.
TOOL_ENTRY_POINT_GROUP = "genfleet.tools"


class LocalToolResolver:
    """
    Resolve against whatever tool providers are installed.

    Deliberately ignores `ref.digest`. Local development is iteration — you are
    editing a tool and an agent together, and refusing to run because the
    working copy does not hash to a published digest would make that
    impossible. `gf agents pull` is the mode that does verify, for when
    the question is "what will the platform actually run".
    """

    def resolve(self, ref: ToolRef) -> Callable:
        installed = discover_installed_tools()
        if ref.slug not in installed:
            known = ", ".join(sorted(installed)) or "none"
            raise ToolResolutionError(
                f"{ref.slug} is not installed. Available: {known}. "
                "Tools come from a package that advertises the "
                f"'{TOOL_ENTRY_POINT_GROUP}' entry point group."
            )
        return installed[ref.slug].load()


def discover_installed_tools() -> dict[str, Any]:
    """
    Every tool advertised by an installed distribution, keyed by slug.

    A provider that fails to load must not hide the others: one broken package
    would otherwise make every tool on the machine unresolvable, and the error
    would name the wrong thing entirely. The failure is logged rather than
    raised, and the tools it would have supplied then fail individually with a
    message naming the tool actually being looked for.
    """
    found: dict[str, Any] = {}
    for entry_point in entry_points(group=TOOL_ENTRY_POINT_GROUP):
        try:
            provider = entry_point.load()
            found.update(provider())
        except Exception as exc:  # noqa: BLE001 — one bad provider, not all
            logger.warning(
                "tool provider %r failed to load and was skipped: %s",
                entry_point.name,
                exc,
            )
    return found




# Set by the engine when it has mounted the agent's tools alongside its code.
# Its presence is what selects the mounted resolver over the local one: agent
# code calls `load_tools()` with no arguments, so the choice cannot be made at
# the call site, and a tenant cannot know where their artifact was mounted.
TOOLS_ROOT_ENV = "GENFLEET_TOOLS_ROOT"

#: The synthetic package every mounted tool is loaded underneath. A namespace
#: rather than a real one — it exists only to give the loaded modules a parent.
_MOUNTED_PACKAGE = "genfleet_mounted_tools"


class MountedToolResolver:
    """
    Resolve against tool artifacts the engine unpacked and mounted read-only.

    The sandbox counterpart to `LocalToolResolver`, and deliberately *only* a
    filesystem lookup. ADR-0010 rejected giving agent code a registry
    credential and network reachability — "spending the isolation ADR-0009
    just bought" — and a resolver that called the registry from inside the
    sandbox would spend it again for tools. So the engine resolves each name
    to a digest on the host, fetches and verifies the bytes there, and mounts
    them; by the time this class runs there is nothing left to authenticate to
    and nothing to fetch.

    That keeps this class inside the [[open-source-boundary]]: it reads
    directories, and knows nothing about the registry that filled them.

    Layout, one directory per tool, named from the slug::

        <root>/genfleet/echo/genfleet.toml   # the tool manifest
        <root>/genfleet/echo/__init__.py     # its code

    `ref.digest` is not re-verified here. The engine verified it before
    unpacking (`sandbox/delivery.verify_digest`), and re-hashing an unpacked
    tree cannot reproduce an archive digest anyway — an unpinned ref reaching
    this far is a bug in the *engine*, which is why `load_tools` reports it
    against the agent rather than silently resolving something.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def resolve(self, ref: ToolRef) -> Callable:
        directory = self._directory_for(ref)
        manifest = self._manifest(ref, directory)
        module = _import_from_directory(ref.slug, directory, manifest.entry)
        _, _, attribute = manifest.entry.partition(":")
        try:
            return getattr(module, attribute)
        except AttributeError as exc:
            raise ToolResolutionError(
                f"{ref.slug}: entry '{manifest.entry}' names '{attribute}', "
                f"which {directory.name} does not define."
            ) from exc

    def _directory_for(self, ref: ToolRef) -> Path:
        # `@scope/name` -> `<root>/scope/name`. The slug is already validated
        # by ToolRef against SLUG_PATTERN, which admits no `..` and no second
        # slash, but the containment check stays: this builds a filesystem
        # path out of a manifest field, and the manifest came from an artifact.
        scope, _, name = ref.slug.removeprefix("@").partition("/")
        directory = self._root / scope / name
        if not directory.resolve().is_relative_to(self._root.resolve()):
            raise ToolResolutionError(f"{ref.slug} resolves outside {self._root}.")
        if not directory.is_dir():
            mounted = ", ".join(sorted(self._mounted_slugs())) or "none"
            raise ToolResolutionError(
                f"{ref.slug} was not mounted. Mounted: {mounted}. The engine "
                "mounts exactly the tools an agent's manifest names, so this "
                "means delivery and the manifest disagree."
            )
        return directory

    def _mounted_slugs(self) -> list[str]:
        """What is actually on the mount, for an error worth reading."""
        if not self._root.is_dir():
            return []
        return [
            f"@{scope.name}/{tool.name}"
            for scope in self._root.iterdir()
            if scope.is_dir()
            for tool in scope.iterdir()
            if tool.is_dir()
        ]

    def _manifest(self, ref: ToolRef, directory: Path) -> ToolManifest:
        """The tool's own manifest, read with the SDK's parser.

        Read from the mount rather than carried in the agent's `ToolRef`: the
        agent names a tool, the tool declares its own entry point, and letting
        the agent's manifest say where a tool's code lives would let one
        artifact redirect another's import.
        """
        try:
            return load_tool_manifest(directory)
        except ManifestError as exc:
            raise ToolResolutionError(
                f"{ref.slug}: the mounted artifact has no usable manifest — {exc}"
            ) from exc


def _import_from_directory(slug: str, directory: Path, entry: str) -> ModuleType:
    """Import a mounted tool's entry module without touching `sys.path`.

    Two constraints shape this. The module in `entry` is written as it would
    be *installed* — `genfleet_tools.echo` — while the mount holds only that
    one tool's directory, so the leading package segments have no counterpart
    on disk. And `sys.path` is process-global: appending each mount would let
    the first tool mounted shadow a later one's imports, which is a
    cross-tool failure with no obvious cause.

    So candidates are the suffixes of the module path, longest first, and the
    module is loaded from its file. Longest first matters: a tool that really
    does ship `genfleet_tools/echo.py` inside its artifact resolves to itself,
    and this stays a widening rather than a change of meaning.
    """
    module_path, separator, attribute = entry.partition(":")
    if not separator or not module_path or not attribute:
        raise ToolResolutionError(
            f"{slug}: entry '{entry}' must be 'module:callable'."
        )
    if module_path.startswith(".") or "/" in module_path or "\\" in module_path:
        raise ToolResolutionError(f"{slug}: entry '{entry}' is not a module path.")

    parts = module_path.split(".")
    tried: list[str] = []
    for start in range(len(parts)):
        suffix = parts[start:]
        for candidate in (
            directory.joinpath(*suffix).with_suffix(".py"),
            directory.joinpath(*suffix) / "__init__.py",
        ):
            if _is_inside(candidate, directory) and candidate.is_file():
                return _load_module(slug, candidate, directory)
            tried.append(str(candidate.relative_to(directory)))

    # The whole module path stripped away: the artifact root *is* the module,
    # which is the common case — `gf tools push` archives the tool's own
    # directory, so `genfleet_tools/echo/__init__.py` arrives as `__init__.py`
    # at the root. Guarded on the directory name matching the last segment so
    # this cannot quietly answer for a differently-named tool.
    root_init = directory / "__init__.py"
    if directory.name == parts[-1] and root_init.is_file():
        return _load_module(slug, root_init, directory)
    tried.append("__init__.py")

    raise ToolResolutionError(
        f"{slug}: entry '{entry}' resolves to none of {', '.join(tried)} "
        f"under the mounted artifact."
    )


def _is_inside(candidate: Path, directory: Path) -> bool:
    """Whether a candidate path stays under the tool's own directory.

    The path is built from the manifest's `entry`, which came out of an
    artifact, so a symlink inside the mount pointing at `/etc` must not become
    an importable module. `resolve()` follows links; `is_relative_to` is what
    catches where they land.
    """
    try:
        return candidate.resolve().is_relative_to(directory.resolve())
    except OSError:  # pragma: no cover - filesystem dependent
        return False


def _load_module(slug: str, file: Path, directory: Path) -> ModuleType:
    """Load `file` under a name derived from the slug.

    Named `genfleet_mounted_tools.<scope>_<name>` rather than the module path
    the entry declares: two tools whose artifacts both contain `tool.py` would
    otherwise claim one `sys.modules` key, and the second import would
    silently return the first tool's module.

    `submodule_search_locations` is set so the loaded module is a *package*,
    which is what makes a tool's own `from . import helpers` work — the
    ordinary way to write a tool of more than one file.
    """
    scope, _, name = slug.removeprefix("@").partition("/")
    module_name = f"{_MOUNTED_PACKAGE}.{scope}_{name}".replace("-", "_")
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    _ensure_mounted_package()

    spec = importlib.util.spec_from_file_location(
        module_name, file, submodule_search_locations=[str(directory)]
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ToolResolutionError(f"{slug}: cannot load a module from {file}.")

    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a tool that imports itself during import
    # would otherwise recurse, and this is third-party code.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_name]
        raise ToolResolutionError(
            f"{slug}: the mounted artifact failed to import — {exc}"
        ) from exc
    return module


def _ensure_mounted_package() -> None:
    """Register the parent package mounted tools are loaded under.

    Python resolves ``from . import helpers`` by importing the *parent* of the
    current module, so a module named ``genfleet_mounted_tools.acme_calc``
    with no ``genfleet_mounted_tools`` in ``sys.modules`` fails with
    ``No module named 'genfleet_mounted_tools'`` — raised from inside the
    tool, which reads as the tool being broken rather than as the loader
    missing a step.

    Created here rather than shipped as a real package on disk: it has no
    contents, and a directory in the SDK named for a runtime detail would
    invite someone to put something in it.
    """
    if _MOUNTED_PACKAGE in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(_MOUNTED_PACKAGE, None, is_package=True)
    # No search locations: the parent must never resolve a submodule by
    # itself. Every real module under it is inserted by `_load_module` from a
    # path the engine mounted, and a parent that could search would let an
    # attribute lookup reach the filesystem on its own terms.
    spec.submodule_search_locations = []
    sys.modules[_MOUNTED_PACKAGE] = importlib.util.module_from_spec(spec)


def default_resolver() -> ToolResolver:
    """The resolver to use when the caller named none.

    Chosen from the environment rather than passed in, because the call that
    needs it is `load_tools()` inside the agent's own `build_agent()` — code
    the platform does not write and cannot ask to pass an argument.
    """
    root = os.environ.get(TOOLS_ROOT_ENV)
    return MountedToolResolver(root) if root else LocalToolResolver()


def load_tools(
    directory: Path | str | None = None,
    *,
    resolver: ToolResolver | None = None,
    manifest: AgentManifest | None = None,
) -> list[Callable]:
    """
    The tools named in an agent's `genfleet.toml`, ready to pass to `Agent`.

        from genfleet.sdk import Agent, load_tools

        def build_agent(remote_tools=()):
            return Agent(
                role="…", model={…}, tools=[*load_tools(), *remote_tools]
            )

    **Take `remote_tools` and merge it.** Two channels reach an agent and
    neither substitutes for the other:

    * `load_tools()` — the tools this agent's *manifest* names, resolved from
      the registry (or from what is installed, locally).
    * `remote_tools=` — what the *runtime* discovered and handed in: other
      agents and remote tool servers reached over A2A, which no manifest can
      name because their URLs are per-deployment.

    The runtime parameter used to be called `tools`, which reads as the first
    channel and is not. Every example agent accepted it and dropped it on
    exactly that misreading, silently losing every remote tool its deployment
    configured — the symptom is a model declining a task it appears to have a
    tool for. `tools=` still works so no published agent breaks; write
    `remote_tools=` in anything new.

    Where the manifest comes from, in order:

    1. **`manifest=`** — the manifest object itself. This is the platform's
       path: it holds the *pinned* manifest from the published artifact, which
       may never touch this machine's filesystem at all.
    2. **`GENFLEET_AGENT_ROOT`** — set by whatever unpacked the agent. It wins
       over `directory` deliberately: after relocation the environment knows
       where the agent actually landed, and a path computed from `__file__`
       when the code was written is a guess that can be stale.
    3. **`directory`**, then the working directory — the local-development
       path, where the manifest really is sitting next to the code.

    `mcps` are *not* included: they are configuration rather than callables and
    reach `Agent` through its own `mcps=` argument.
    """
    if manifest is None:
        root = os.environ.get(AGENT_ROOT_ENV) or directory or "."
        spec = load_agent_manifest(root)
    else:
        spec = manifest
    active = resolver or default_resolver()

    resolved: list[Callable] = []
    for ref in spec.tools:
        try:
            resolved.append(active.resolve(ref))
        except ToolResolutionError as exc:
            # Name the agent as well as the tool. With several agents in one
            # repo, "not installed" alone does not say which manifest to fix.
            raise ToolResolutionError(f"{spec.name}: {exc}") from exc
    return resolved
