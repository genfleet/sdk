"""Manifests are hand-authored, so the errors matter as much as the parsing."""

from __future__ import annotations

import re

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
    with pytest.raises(ManifestError, match="agent.toml"):
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


def test_an_agent_says_whether_its_model_can_be_chosen(tmp_path):
    _write(tmp_path, '[agent]\nname = "support"\nmodel_choice = true\n')
    assert load_agent_manifest(tmp_path).model_choice is True
    _write(tmp_path, '[agent]\nname = "support"\n')
    assert load_agent_manifest(tmp_path).model_choice is False


def test_model_choice_must_be_a_boolean(tmp_path):
    with pytest.raises(ManifestError, match="model_choice"):
        load_agent_manifest(_write(tmp_path, '[agent]\nname = "support"\nmodel_choice = "yes"\n'))


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


def test_content_data_defaults_to_empty(tmp_path):
    manifest = load_agent_manifest(_write(tmp_path, '[agent]\nname = "a"\n[content]\nrequired = true\n'))
    assert manifest.content.data == []


def test_content_data_globs_are_read(tmp_path):
    manifest = load_agent_manifest(
        _write(tmp_path, '[agent]\nname = "a"\n[content]\ndata = ["knowledge.json", "flows/**", "**/*.toml"]\n')
    )
    assert manifest.content.data == ["knowledge.json", "flows/**", "**/*.toml"]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ('"knowledge.json"', "'data' must be a list of glob strings"),
        ("[1]", "'data' must be a list of glob strings"),
        ('["/etc/passwd"]', "relative to the content root"),
        ('["../x"]', "'..'"),
        ('["a//b"]', "empty"),
        ('["a\\\\b"]', "backslash"),
        ('["a/**b"]', "whole folder segment"),
        ('["{a,b}.json"]', "brackets or braces"),
        ('[""]', "non-empty"),
        ('["a", "a"]', "twice"),
    ],
)
def test_a_bad_content_data_glob_is_refused(tmp_path, data, message):
    with pytest.raises(ManifestError, match=re.escape(message)):
        load_agent_manifest(_write(tmp_path, f'[agent]\nname = "a"\n[content]\ndata = {data}\n'))


def test_too_many_content_data_globs_are_refused(tmp_path):
    globs = ", ".join(f'"f{i}.json"' for i in range(101))
    with pytest.raises(ManifestError, match="more than 100 globs"):
        load_agent_manifest(_write(tmp_path, f'[agent]\nname = "a"\n[content]\ndata = [{globs}]\n'))


# ── agent.toml, [data], [connectors], [settings] (0.28.0) ───────────────────

def _agent_dir(tmp_path, body: str, name: str = "agent.toml"):
    (tmp_path / name).write_text(body)
    return tmp_path


def test_agent_toml_is_read(tmp_path):
    manifest = load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n'))
    assert manifest.name == "a"


def test_legacy_name_is_still_read(tmp_path):
    manifest = load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n', "genfleet.toml"))
    assert manifest.name == "a"


def test_both_manifest_names_is_an_error(tmp_path):
    _agent_dir(tmp_path, '[agent]\nname = "a"\n')
    _agent_dir(tmp_path, '[agent]\nname = "a"\n', "genfleet.toml")
    with pytest.raises(ManifestError, match="keep only agent.toml"):
        load_agent_manifest(tmp_path)


def test_no_manifest_names_agent_toml(tmp_path):
    with pytest.raises(ManifestError, match="no agent.toml here"):
        load_agent_manifest(tmp_path)


def test_data_table_replaces_content(tmp_path):
    manifest = load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n[data]\nrequired = true\ndata = ["*.toml"]\n'))
    assert manifest.content.required is True and manifest.content.data == ["*.toml"]


def test_data_and_content_together_is_an_error(tmp_path):
    with pytest.raises(ManifestError, match=r"both \[data\] and \[content\]"):
        load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n[data]\nrequired = true\n[content]\nrequired = true\n'))


def test_data_errors_name_the_data_table(tmp_path):
    with pytest.raises(ManifestError, match=r"\[data\] unknown key 'requried'"):
        load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n[data]\nrequried = true\n'))


_FULL = '''
[agent]
name = "sales"

[connectors.erp]
provider = "custom_api"
description = "The ERP that receives leads."

[connectors.crm]
provider = "custom_api"
required = false

[settings.lead_sink]
type = "enum"
options = ["erp", "views"]
default = "views"
label = "Where leads go"
group = "Leads"

[settings.open_lead_window_days]
type = "integer"
min = 0
max = 365
default = 30

[settings.source_map]
type = "map"
default = { whatsapp = "WhatsApp" }

[settings.tags]
type = "list"

[settings.strict]
type = "boolean"
required = true
'''


def test_connectors_and_settings_parse(tmp_path):
    manifest = load_agent_manifest(_agent_dir(tmp_path, _FULL))
    assert manifest.connectors["erp"].provider == "custom_api" and manifest.connectors["erp"].required is True
    assert manifest.connectors["crm"].required is False
    assert manifest.settings["lead_sink"].options == ["erp", "views"]
    assert manifest.settings["open_lead_window_days"].default == 30
    assert manifest.settings["source_map"].default == {"whatsapp": "WhatsApp"}
    assert manifest.settings["strict"].required is True


@pytest.mark.parametrize(
    "body, message",
    [
        ('[connectors.ERP]\nprovider = "custom_api"\n', r"\[connectors.ERP\]: the name"),
        ('[connectors.erp]\n', r"\[connectors.erp\]: provider"),
        ('[connectors.erp]\nprovider = "Custom API"\n', "connector id"),
        ('[connectors.erp]\nprovider = "custom_api"\nsecret = "x"\n', "unknown key 'secret'"),
        ('[connectors]\nerp = "custom_api"\n', r"\[connectors.erp\] must be a table"),
        ('[settings.a]\ntype = "secret"\n', r"\[settings.a\]: type"),
        ('[settings.a]\ntype = "enum"\n', "an enum needs 'options'"),
        ('[settings.a]\ntype = "enum"\noptions = ["x", "x"]\n', "an enum needs 'options'"),
        ('[settings.a]\noptions = ["x"]\n', "'options' is only for"),
        ('[settings.a]\nmin = 1\n', "'min' and 'max' are only for"),
        ('[settings.a]\ntype = "integer"\nmin = 5\nmax = 1\n', "'min' is above 'max'"),
        ('[settings.a]\ntype = "integer"\ndefault = 1.5\n', "'default' must be a whole number"),
        ('[settings.a]\ntype = "integer"\ndefault = true\n', "'default' must be a whole number"),
        ('[settings.a]\ntype = "integer"\nmax = 3\ndefault = 4\n', "'default' must be at most 3"),
        ('[settings.a]\ntype = "enum"\noptions = ["x"]\ndefault = "y"\n', "'default' must be one of 'x'"),
        ('[settings.a]\ntype = "map"\ndefault = { k = 1 }\n', "'default' must be a table of strings"),
        ('[settings.a]\ntype = "list"\ndefault = [1]\n', "'default' must be a list of strings"),
        ('[settings.a]\ntype = "boolean"\ndefault = "yes"\n', "'default' must be true or false"),
        ('[settings.a]\ndefault = 3\n', "'default' must be a string"),
        ('[settings.a]\nvalue = "x"\n', "unknown key 'value'"),
    ],
)
def test_bad_connectors_and_settings_are_refused(tmp_path, body, message):
    with pytest.raises(ManifestError, match=message):
        load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n' + body))


def test_too_many_slots_and_settings(tmp_path):
    slots = "".join(f'[connectors.s{i}]\nprovider = "custom_api"\n' for i in range(9))
    with pytest.raises(ManifestError, match="more than 8"):
        load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n' + slots))
    keys = "".join(f"[settings.k{i}]\n" for i in range(41))
    with pytest.raises(ManifestError, match="more than 40"):
        load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n' + keys))


def test_setting_value_problem_limits():
    from genfleet.sdk.manifest import SettingSpec, setting_value_problem

    assert setting_value_problem(SettingSpec(), "x" * 500) is None
    assert setting_value_problem(SettingSpec(), "x" * 501) == "must be at most 500 characters"
    assert setting_value_problem(SettingSpec(type="text"), "x" * 20_000) is None
    assert setting_value_problem(SettingSpec(type="number"), 1.5) is None
    assert setting_value_problem(SettingSpec(type="number"), True) == "must be a number"
    assert setting_value_problem(SettingSpec(type="list"), ["a"] * 201) is not None
    assert setting_value_problem(SettingSpec(type="map"), {"k": "v"}) is None


@pytest.mark.parametrize(
    "body, message",
    [
        ('[settings."a\\n"]\n', r"the name"),
        ('[connectors.erp]\nprovider = "custom_api\\n"\n', "connector id"),
        ('[settings.a]\ntype = "integer"\nmin = "5"\n', r"\[settings.a\]: min"),
        ('[settings.a]\ntype = "integer"\nmin = true\n', r"\[settings.a\]: min"),
        ('[settings.a]\ntype = "integer"\nmax = 2.5\n', "'max' of an integer setting must be a whole number"),
        ('[settings.a]\ntype = "number"\nmax = inf\n', "'max' must be a finite number"),
        ('[settings.a]\ntype = "number"\ndefault = nan\n', "'default' must be a number"),
        ('[settings.a]\ntype = "number"\ndefault = inf\n', "'default' must be a number"),
        ('[settings.a]\ntype = "integer"\nmin = 1000000\ndefault = 1\n', "'default' must be at least 1000000"),
        ('[settings.a]\ntype = "number"\nmin = 0.5\ndefault = 0.1\n', "'default' must be at least 0.5"),
        ('[settings.a]\ntype = "enum"\noptions = [""]\n', "each of 'options'"),
        ('[settings.a]\ntype = "enum"\noptions = ["' + "x" * 501 + '"]\n', "each of 'options'"),
    ],
)
def test_loose_setting_values_are_refused(tmp_path, body, message):
    with pytest.raises(ManifestError, match=message):
        load_agent_manifest(_agent_dir(tmp_path, '[agent]\nname = "a"\n' + body))


@pytest.mark.parametrize("key", ["connectors", "settings", "tools", "views", "content", "mcps"])
def test_a_table_written_inside_agent_is_a_manifest_error(tmp_path, key):
    with pytest.raises(ManifestError, match=rf"\[agent\] has '{key}', which is its own table"):
        load_agent_manifest(_agent_dir(tmp_path, f'[agent]\nname = "a"\n{key} = 1\n'))


def test_agent_manifest_path(tmp_path):
    from genfleet.sdk.manifest import agent_manifest_path

    assert agent_manifest_path(tmp_path).name == "agent.toml"  # neither: the new name, for the error
    (tmp_path / "genfleet.toml").write_text("")
    assert agent_manifest_path(tmp_path).name == "genfleet.toml"
    (tmp_path / "agent.toml").write_text("")
    with pytest.raises(ManifestError, match="keep only agent.toml"):
        agent_manifest_path(tmp_path)
    (tmp_path / "genfleet.toml").unlink()
    assert agent_manifest_path(tmp_path).name == "agent.toml"
