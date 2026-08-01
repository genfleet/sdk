"""Builds the wheel and asserts its archive layout.

`tests/test_packaging.py` guards the *source tree* and the *installed*
interpreter state. Neither catches the two ways the build config can ship a
broken wheel from a source tree that looks correct:

1. Hatchling is pointed at `packages = ["agentinc"]` (or the source tree grows
   an `agentinc/__init__.py`). The wheel then contains `agentinc/__init__.py`,
   which turns the shared `agentinc` namespace into a *regular* package on the
   consumer's machine and shadows every sibling distribution
   (agentinc-core, agentinc-engine, ...). This is the regression that shipped.

2. The naive fix `packages = ["agentinc/sdk"]` is applied. Hatchling strips the
   parent and emits a wheel with a **top-level `sdk/` package** — importable as
   `import sdk`, never as `agentinc.sdk`. It builds with no warning at all.

Only inspecting the built archive catches either one, so this module actually
runs the build backend and reads the resulting zip.
"""

import glob
import os
import subprocess
import sys
import zipfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_wheel(outdir: str) -> str:
    """Build a wheel into `outdir`, return its path. Skip if no builder."""
    builders = [
        [sys.executable, "-m", "build", "--wheel", "--outdir", outdir, REPO_ROOT],
        ["uv", "build", "--wheel", "--out-dir", outdir, REPO_ROOT],
    ]
    errors = []
    for cmd in builders:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{cmd[0]}: {exc}")
            continue
        if proc.returncode == 0:
            break
        errors.append(f"{' '.join(map(str, cmd))} failed:\n{proc.stdout}\n{proc.stderr}")
    else:
        pytest.skip(
            "no working wheel builder available "
            "(install the `dev` extra for `build`, or install uv):\n"
            + "\n".join(errors)
        )

    wheels = glob.glob(os.path.join(outdir, "*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory) -> list[str]:
    outdir = str(tmp_path_factory.mktemp("wheel"))
    with zipfile.ZipFile(_build_wheel(outdir)) as zf:
        return zf.namelist()


@pytest.mark.slow
def test_wheel_has_no_namespace_init(wheel_names):
    """Failure mode 1: an `agentinc/__init__.py` in the wheel."""
    assert "agentinc/__init__.py" not in wheel_names, (
        "wheel ships agentinc/__init__.py — this makes `agentinc` a regular "
        "package and shadows every sibling agentinc-* distribution. "
        "Keep `only-include = [\"agentinc/sdk\"]` in pyproject.toml."
    )


@pytest.mark.slow
def test_wheel_ships_the_sdk_subpackage(wheel_names):
    """The wheel must actually contain the code, under the right prefix."""
    assert "agentinc/sdk/__init__.py" in wheel_names, (
        f"wheel is missing agentinc/sdk/__init__.py; contents: {wheel_names}"
    )


@pytest.mark.slow
def test_wheel_has_no_top_level_sdk_package(wheel_names):
    """Failure mode 2: `packages = ["agentinc/sdk"]` flattens to top-level `sdk/`."""
    flattened = [n for n in wheel_names if n.startswith("sdk/")]
    assert not flattened, (
        f"wheel contains a top-level sdk/ package: {flattened}. "
        "hatchling's `packages` strips the parent directory; use "
        "`only-include` so the `agentinc/` prefix is preserved."
    )


@pytest.mark.slow
def test_wheel_top_level_is_only_agentinc_and_metadata(wheel_names):
    """Nothing else leaks into the wheel root."""
    tops = {n.split("/")[0] for n in wheel_names}
    unexpected = {t for t in tops if t != "agentinc" and not t.endswith(".dist-info")}
    assert not unexpected, f"unexpected top-level entries in wheel: {sorted(unexpected)}"
