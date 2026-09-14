"""Manifests are hand-authored, so the errors matter as much as the parsing."""

from __future__ import annotations

import pytest

from genfleet.sdk.manifest import (
    ManifestError,
    load_agent_manifest,
    load_tool_manifest,
)


def _write(directory, body: str):
    (directory / "genfleet.toml").write_text(body)
    return directory


def test_agent_names_tools_without_versions_or_hashes(tmp_path):
    # The point of ADR-0007's revision: a developer writes names, never digests.
    manifest = load_agent_manifest(
        _write(
            tmp_path,
            """
            [agent]
            name = "support-triage"

            [tools]
            "@acme/jira" = "*"
            "@acme/slack" = { spec = "2.0.x" }

            [mcps]
            "@acme/gdrive" = "*"
            """,
        )
    )

    assert [t.slug for t in manifest.tools] == ["@acme/jira", "@acme/slack"]
    assert manifest.tools[1].spec == "2.0.x"
    assert [m.slug for m in manifest.mcps] == ["@acme/gdrive"]


def test_unresolved_manifest_is_not_publishable(tmp_path):
    # Publishing must resolve every ref to a version and digest. A manifest
    # straight off disk never is, and `push` relies on that being detectable.
    manifest = load_agent_manifest(
        _write(tmp_path, '[agent]\nname = "a"\n\n[tools]\n"@acme/jira" = "*"\n')
    )
    assert not manifest.is_fully_pinned

    manifest.tools[0].version = "1.4.2"
    manifest.tools[0].digest = "sha256:ab12"
    assert manifest.is_fully_pinned


def test_agent_with_no_tools_is_fully_pinned(tmp_path):
    # Vacuous truth, and load-bearing: an agent that needs nothing must not be
    # blocked from publishing by a resolution step with nothing to resolve.
    manifest = load_agent_manifest(_write(tmp_path, '[agent]\nname = "solo"\n'))
    assert manifest.is_fully_pinned


@pytest.mark.parametrize(
    "slug",
    ["jira", "@acme", "acme/jira", "@Acme/jira", "@acme/", "@/jira"],
    ids=["no-scope", "no-name", "no-sigil", "uppercase", "empty-name", "empty-scope"],
)
def test_malformed_slug_is_rejected(tmp_path, slug):
    with pytest.raises(ManifestError, match="tool slug"):
        load_agent_manifest(
            _write(tmp_path, f'[agent]\nname = "a"\n\n[tools]\n"{slug}" = "*"\n')
        )


def test_secret_value_in_a_tool_manifest_is_refused(tmp_path):
    # A manifest is committed and published to a registry every tenant can
    # read. A value here is a leak when written, not when used.
    with pytest.raises(ManifestError, match="declare secret names only"):
        load_tool_manifest(
            _write(
                tmp_path,
                """
                [tool]
                slug = "@acme/jira"
                version = "1.0.0"
                entry = "genfleet_tools.jira:search"

                [secrets]
                api_token = { value = "sk-live-oops" }
                """,
            )
        )


def test_tool_declares_config_and_secret_names(tmp_path):
    manifest = load_tool_manifest(
        _write(
            tmp_path,
            """
            [tool]
            slug = "@acme/jira"
            version = "1.4.2"
            kind = "cli"
            entry = "genfleet_tools.jira:search_issues"

            [config]
            base_url = { type = "string", required = true }

            [secrets]
            api_token = { description = "Jira API token" }
            """,
        )
    )

    assert manifest.config["base_url"].required
    assert manifest.secrets["api_token"].required
    assert manifest.entry == "genfleet_tools.jira:search_issues"


def test_missing_manifest_names_the_path(tmp_path):
    # These are authored by hand; an error that does not say where is useless.
    with pytest.raises(ManifestError, match="genfleet.toml"):
        load_agent_manifest(tmp_path)


def test_malformed_toml_names_the_path(tmp_path):
    with pytest.raises(ManifestError, match="not valid TOML"):
        load_agent_manifest(_write(tmp_path, "[agent\nname ="))


def test_manifest_without_its_required_section_is_rejected(tmp_path):
    with pytest.raises(ManifestError, match=r"missing an \[agent\] section"):
        load_agent_manifest(_write(tmp_path, '[tool]\nslug = "@acme/x"\n'))
