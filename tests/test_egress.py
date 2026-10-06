"""ADR-0021 §3: the `egress` entries an agent or tool declares.

Driven by tests/fixtures/egress-hosts.cases.json, which the backend copies so
its DTO validator and this parser cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genfleet.sdk.egress import (
    MAX_EGRESS_ENTRIES,
    MAX_HOST_LENGTH,
    EgressEntry,
    EgressEntryError,
    parse_egress,
    parse_egress_entry,
)
from genfleet.sdk.manifest import ManifestError, load_agent_manifest, load_tool_manifest

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "egress-hosts.cases.json").read_text()
)


@pytest.mark.parametrize("case", CASES["entries"], ids=lambda c: repr(c["input"]))
def test_entry_syntax(case: dict) -> None:
    if case["valid"]:
        assert parse_egress_entry(case["input"]).canonical == case["canonical"]
    else:
        with pytest.raises(EgressEntryError) as exc:
            parse_egress_entry(case["input"])
        assert exc.value.reason == case["reason"]


@pytest.mark.parametrize(
    "case", CASES["matches"], ids=lambda c: f"{c['pattern']} ~ {c['host']}:{c['port']}"
)
def test_matching(case: dict) -> None:
    entry = parse_egress_entry(case["pattern"])
    assert entry.matches(case["host"], case["port"]) is case["matches"]


def test_the_limits_in_the_cases_file_are_the_parsers() -> None:
    assert CASES["limits"] == {
        "max_entries": MAX_EGRESS_ENTRIES,
        "max_host_length": MAX_HOST_LENGTH,
    }


def test_a_host_longer_than_253_characters_is_refused() -> None:
    labels = ["a" * 63] * 4  # 4 * 63 + 3 dots = 255
    with pytest.raises(EgressEntryError) as exc:
        parse_egress_entry(".".join(labels))
    assert exc.value.reason == "invalid_host"
    parse_egress_entry(".".join(["a" * 63] * 3 + ["a" * 61]))  # exactly 253


def test_a_list_is_capped_and_deduplicated_by_canonical_form() -> None:
    assert parse_egress(["API.github.com", "api.github.com:443", "x.example.com"]) == [
        EgressEntry(host="api.github.com", port=443),
        EgressEntry(host="x.example.com", port=443),
    ]
    with pytest.raises(EgressEntryError) as exc:
        parse_egress([f"h{i}.example.com" for i in range(MAX_EGRESS_ENTRIES + 1)])
    assert exc.value.reason == "limit"


def _write(directory: Path, body: str) -> Path:
    (directory / "genfleet.toml").write_text(body)
    return directory


def test_an_agent_manifest_declares_egress_in_canonical_form(tmp_path: Path) -> None:
    manifest = load_agent_manifest(
        _write(
            tmp_path,
            """
            [agent]
            name = "support-triage"
            egress = ["API.github.com:443", "*.googleapis.com"]
            """,
        )
    )
    assert manifest.egress == ["api.github.com", "*.googleapis.com"]


def test_a_tool_manifest_declares_egress(tmp_path: Path) -> None:
    manifest = load_tool_manifest(
        _write(
            tmp_path,
            """
            [tool]
            slug = "@acme/weather"
            version = "1.0.0"
            entry = "weather:run"
            egress = ["api.open-meteo.com"]
            """,
        )
    )
    assert manifest.egress == ["api.open-meteo.com"]


def test_egress_is_optional(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_write(tmp_path, '[agent]\nname = "x"\n'))
    assert manifest.egress == []


def test_a_bad_entry_names_the_file_and_the_entry(tmp_path: Path) -> None:
    with pytest.raises(ManifestError) as exc:
        load_agent_manifest(
            _write(tmp_path, '[agent]\nname = "x"\negress = ["10.0.0.1"]\n')
        )
    assert "genfleet.toml" in str(exc.value)
    assert "10.0.0.1" in str(exc.value)
