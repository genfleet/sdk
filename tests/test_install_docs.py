"""Every install command the docs show must be one a stranger can run.

The package is not on PyPI (sdk#12), so a bare `pip install genfleet-sdk[...]`
fails for every new developer. The docs said exactly that in the hero of
`docs/index.html` while the README had already moved to the git URL.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "docs" / "index.html", *sorted((ROOT / "skills").rglob("*.md"))]

# `pip install` / `uv add` followed, on the same line, by a genfleet-sdk requirement.
INSTALL = re.compile(r"(?:pip install|uv add|uv pip install)[^\n`<]*?genfleet-sdk[^\s'\"`<]*[^\n`<]*")


def test_every_install_command_names_the_git_source():
    offenders = []
    for path in DOCS:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for match in INSTALL.finditer(line):
                if "git+https://github.com/genfleet/sdk" not in match.group(0):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {match.group(0).strip()}")
    assert not offenders, "install commands that assume PyPI:\n" + "\n".join(offenders)
