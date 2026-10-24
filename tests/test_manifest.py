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
    # The point of the manifest design: a developer writes names, never digests.
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


def test_an_agent_declares_the_catalog_models_it_calls(tmp_path):
    _write(tmp_path, '[agent]\nname = "support"\nmodels = ["catalog-model", "catalog-model", "catalog-model-2"]\n')
    # Platform model ids, deduplicated and lowercased; the platform scopes the
    # agent's model credential to these.
    assert load_agent_manifest(tmp_path).models == ["catalog-model", "catalog-model-2"]


def test_an_agent_without_models_declares_none(tmp_path):
    _write(tmp_path, '[agent]\nname = "support"\n')
    assert load_agent_manifest(tmp_path).models == []


@pytest.mark.parametrize("model", ["openai/gpt-4o", "", "-fast", "a", "x" * 65, "general fast"])
def test_a_model_entry_must_be_a_plain_catalog_id(tmp_path, model):
    _write(tmp_path, f'[agent]\nname = "support"\nmodels = ["{model}"]\n')
    with pytest.raises(ManifestError, match="models"):
        load_agent_manifest(tmp_path)


# A manifest written for an older SDK may still carry an `egress` list. The
# SDK no longer reads it: the key loads, is ignored, and never fails, even
# when its value would once have been refused.
@pytest.mark.parametrize("egress", ['["api.github.com"]', '["10.0.0.1"]', '"not-a-list"'])
def test_an_agent_manifest_ignores_an_egress_key(tmp_path, egress):
    _write(tmp_path, f'[agent]\nname = "support"\negress = {egress}\n')
    manifest = load_agent_manifest(tmp_path)
    assert manifest.name == "support"
    assert not hasattr(manifest, "egress")
    assert "egress" not in manifest.model_dump()


@pytest.mark.parametrize("egress", ['["api.open-meteo.com"]', '["10.0.0.1"]', '"not-a-list"'])
def test_a_tool_manifest_ignores_an_egress_key(tmp_path, egress):
    _write(
        tmp_path,
        f'[tool]\nslug = "@acme/weather"\nversion = "1.0.0"\n'
        f'entry = "weather:forecast"\negress = {egress}\n',
    )
    manifest = load_tool_manifest(tmp_path)
    assert manifest.slug == "@acme/weather"
    assert not hasattr(manifest, "egress")
    assert "egress" not in manifest.model_dump()


@pytest.mark.parametrize("section", ["tools", "mcps"])
def test_a_tool_ref_ignores_an_egress_key(tmp_path, section):
    _write(
        tmp_path,
        f'[agent]\nname = "support"\n\n[{section}]\n'
        '"@acme/x" = { spec = "1.x", egress = ["10.0.0.1"] }\n',
    )
    manifest = load_agent_manifest(tmp_path)
    (ref,) = getattr(manifest, section)
    assert (ref.slug, ref.spec) == ("@acme/x", "1.x")
    assert not hasattr(ref, "egress")
    assert "egress" not in ref.model_dump()


def test_content_table_defaults_to_not_required(tmp_path):
    manifest = load_agent_manifest(_write(tmp_path, '[agent]\nname = "a"\n'))
    assert manifest.content.required is False


def test_content_required_is_read(tmp_path):
    manifest = load_agent_manifest(_write(tmp_path, '[agent]\nname = "a"\n[content]\nrequired = true\n'))
    assert manifest.content.required is True


def test_content_required_false_is_read(tmp_path):
    manifest = load_agent_manifest(_write(tmp_path, '[agent]\nname = "a"\n[content]\nrequired = false\n'))
    assert manifest.content.required is False


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ('[content]\nrequired = "yes"\n', "'required' must be true or false"),
        ("[content]\nrequired = 1\n", "'required' must be true or false"),
        ("[content]\nrequred = true\n", "unknown key 'requred'"),
    ],
)
def test_a_bad_content_table_is_refused(tmp_path, table, message):
    with pytest.raises(ManifestError, match=message):
        load_agent_manifest(_write(tmp_path, '[agent]\nname = "a"\n' + table))


def test_content_that_is_not_a_table_is_refused(tmp_path):
    with pytest.raises(ManifestError, match="must be a table"):
        load_agent_manifest(_write(tmp_path, 'content = true\n[agent]\nname = "a"\n'))


def test_a_section_error_names_the_file_once(tmp_path):
    with pytest.raises(ManifestError) as raised:
        load_agent_manifest(_write(tmp_path, '[agent]\nname = "a"\n[content]\nrequired = "yes"\n'))
    assert str(raised.value).count("genfleet.toml") == 1
