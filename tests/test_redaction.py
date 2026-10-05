"""Redaction v2 (tool-call stream contract v1.1 §C) — the same cases as the
engine's and the backend's table-driven tests."""

from __future__ import annotations

import pytest

from genfleet.sdk._redaction import (
    ARGUMENTS_MAX_BYTES,
    ARGUMENTS_PREVIEW_CHARS,
    REDACTED,
    redact,
    safe_arguments,
)

CASES = [
    # Rule 1: a key part, or the compact key, names a credential.
    ({"api_key": "v"}, {"api_key": REDACTED}),
    ({"apiKey": "v"}, {"apiKey": REDACTED}),
    ({"x-auth-token": "v"}, {"x-auth-token": REDACTED}),
    ({"githubToken": "v"}, {"githubToken": REDACTED}),
    ({"pwd": "v"}, {"pwd": REDACTED}),
    ({"Cookie": "v"}, {"Cookie": REDACTED}),
    ({"client.secret": "v"}, {"client.secret": REDACTED}),
    ({"private_key": "v"}, {"private_key": REDACTED}),
    ({"accesstoken": "v"}, {"accesstoken": REDACTED}),
    ({"dbpassword": "v"}, {"dbpassword": REDACTED}),
    ({"stripeapikey": "v"}, {"stripeapikey": REDACTED}),
    ({"sshprivatekey": "v"}, {"sshprivatekey": REDACTED}),
    # Kept: no part names a credential and no suffix matches.
    ({"keyword": "v"}, {"keyword": "v"}),
    ({"monkey": "v"}, {"monkey": "v"}),
    ({"keys": "v"}, {"keys": "v"}),
    ({"author": "v"}, {"author": "v"}),
    ({"tokenizer": "v"}, {"tokenizer": "v"}),
    ({"session_id": "v"}, {"session_id": "v"}),
    ({"session": "v"}, {"session": "v"}),
    # Rule 2: header-style pairs.
    ({"name": "Authorization", "value": "v"}, {"name": "Authorization", "value": REDACTED}),
    ({"header": "X-Api-Key", "value": "v"}, {"header": "X-Api-Key", "value": REDACTED}),
    ({"name": "Accept", "value": "json"}, {"name": "Accept", "value": "json"}),
    # Rule 3: values that look like credentials, whatever the key.
    ({"h": "Bearer abc"}, {"h": REDACTED}),
    ({"h": "Basic dXNlcjpwdw=="}, {"h": REDACTED}),
    ({"v": "sk-proj-123"}, {"v": REDACTED}),
    ({"v": "ghp_abc"}, {"v": REDACTED}),
    ({"v": "github_pat_abc"}, {"v": REDACTED}),
    ({"v": "xoxb-1-2"}, {"v": REDACTED}),
    ({"v": "AKIAABCDEFGHIJKLMNOP"}, {"v": REDACTED}),
    ({"v": "-----BEGIN RSA PRIVATE KEY-----\nMII"}, {"v": REDACTED}),
    ({"v": "a plain sentence"}, {"v": "a plain sentence"}),
    # Every depth.
    ({"a": [{"b": {"password": "v"}}]}, {"a": [{"b": {"password": REDACTED}}]}),
]


@pytest.mark.parametrize(("value", "expected"), CASES)
def test_redaction_v2(value: dict, expected: dict) -> None:
    assert redact(value) == expected


def test_arguments_over_the_cap_become_a_truncation_marker() -> None:
    capped = safe_arguments({"content": "x" * ARGUMENTS_MAX_BYTES})

    assert capped["_truncated"] is True
    assert len(capped["preview"]) == ARGUMENTS_PREVIEW_CHARS


def test_the_cap_applies_after_redaction() -> None:
    capped = safe_arguments({"token": "x" * ARGUMENTS_MAX_BYTES})

    assert capped == {"token": REDACTED}
