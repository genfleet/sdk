"""Tests for the Gemini provider's OpenAI-format → Gemini contents conversion."""

from __future__ import annotations

import base64
import json

import pytest

from agentinc.sdk.agent import _tool_call_to_dict
from agentinc.sdk.providers.gemini import (
    _decode_signature,
    _to_gemini_messages,
    _tool_calls_from_response,
)
from agentinc.sdk.providers.openai import _sanitize_messages
from agentinc.sdk.schemas import Message, ToolCall


def test_system_and_user_messages():
    system, contents = _to_gemini_messages([
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ])
    assert system == "be helpful"
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_assistant_text():
    _, contents = _to_gemini_messages([
        {"role": "assistant", "content": "hello"},
    ])
    assert contents == [{"role": "model", "parts": [{"text": "hello"}]}]


def test_assistant_tool_calls_become_function_call_parts():
    _, contents = _to_gemini_messages([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result!"},
    ])
    assert contents == [
        {
            "role": "model",
            "parts": [{"function_call": {"name": "lookup", "args": {"q": "x"}}}],
        },
        {
            "role": "user",
            "parts": [{"function_response": {
                "name": "lookup",
                "response": {"result": "result!"},
            }}],
        },
    ]


def test_assistant_text_alongside_tool_calls():
    _, contents = _to_gemini_messages([
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "lookup", "arguments": "{}"}},
            ],
        },
    ])
    assert contents[0]["parts"] == [
        {"text": "let me check"},
        {"function_call": {"name": "lookup", "args": {}}},
    ]


def test_malformed_arguments_fall_back_to_empty():
    _, contents = _to_gemini_messages([
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "lookup", "arguments": "not json"}},
            ],
        },
    ])
    assert contents[0]["parts"] == [{"function_call": {"name": "lookup", "args": {}}}]


def test_empty_assistant_message_skipped():
    _, contents = _to_gemini_messages([
        {"role": "assistant", "content": ""},
    ])
    assert contents == []


# ---------------------------------------------------------------------------
# thought_signature — Gemini 3.x requires each replayed function_call to carry
# the opaque signature it was issued with, on the enclosing Part.
# ---------------------------------------------------------------------------

SIG = b"\x00\x01\xfe\xff signature bytes"
SIG_B64 = base64.b64encode(SIG).decode("ascii")


class _FakeFunctionCall:
    def __init__(self, name, args, id=None):
        self.name = name
        self.args = args
        self.id = id


class _FakePart:
    def __init__(self, function_call=None, thought_signature=None):
        self.function_call = function_call
        self.thought_signature = thought_signature


class _FakeResponse:
    """Minimal stand-in for a genai response/chunk: candidates → content → parts."""

    def __init__(self, parts):
        content = type("C", (), {"parts": parts})()
        self.candidates = [type("Cand", (), {"content": content})()]


def _assistant_turn(*tool_calls):
    return {"role": "assistant", "content": "", "tool_calls": list(tool_calls)}


def test_signature_captured_from_part_not_function_call():
    response = _FakeResponse([
        _FakePart(_FakeFunctionCall("lookup", {"q": "x"}, id="t1"), thought_signature=SIG),
    ])
    calls = _tool_calls_from_response(response)
    assert calls == [ToolCall(
        id="t1", name="lookup", arguments={"q": "x"}, thought_signature=SIG_B64,
    )]


def test_signature_absent_leaves_field_none():
    response = _FakeResponse([_FakePart(_FakeFunctionCall("lookup", {}))])
    calls = _tool_calls_from_response(response)
    assert calls[0].thought_signature is None
    assert calls[0].id == "lookup"  # falls back to name when the API sends no id


def test_text_parts_ignored_when_collecting_tool_calls():
    response = _FakeResponse([
        _FakePart(None, thought_signature=b"text-part-sig"),
        _FakePart(_FakeFunctionCall("lookup", {}), thought_signature=SIG),
    ])
    calls = _tool_calls_from_response(response)
    assert len(calls) == 1
    assert calls[0].thought_signature == SIG_B64


def test_multiple_tool_calls_keep_their_own_signatures():
    sig_b = base64.b64encode(b"second").decode("ascii")
    response = _FakeResponse([
        _FakePart(_FakeFunctionCall("a", {}, id="t1"), thought_signature=SIG),
        _FakePart(_FakeFunctionCall("b", {}, id="t2"), thought_signature=b"second"),
    ])
    calls = _tool_calls_from_response(response)
    assert [(c.id, c.thought_signature) for c in calls] == [("t1", SIG_B64), ("t2", sig_b)]


def test_signature_replayed_on_part_as_bytes():
    _, contents = _to_gemini_messages([
        _assistant_turn({
            "id": "t1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q": "x"}'},
            "thought_signature": SIG_B64,
        }),
    ])
    assert contents[0]["parts"] == [{
        "function_call": {"name": "lookup", "args": {"q": "x"}},
        "thought_signature": SIG,
    }]


def test_multiple_signatures_replayed_independently():
    _, contents = _to_gemini_messages([
        _assistant_turn(
            {"id": "t1", "type": "function", "function": {"name": "a", "arguments": "{}"},
             "thought_signature": SIG_B64},
            {"id": "t2", "type": "function", "function": {"name": "b", "arguments": "{}"},
             "thought_signature": base64.b64encode(b"second").decode("ascii")},
        ),
    ])
    assert [p["thought_signature"] for p in contents[0]["parts"]] == [SIG, b"second"]


def test_history_without_signature_still_converts():
    """Rows written before signatures were captured must not start erroring."""
    for tc in (
        {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"},
         "thought_signature": None},
    ):
        _, contents = _to_gemini_messages([_assistant_turn(tc)])
        assert contents[0]["parts"] == [{"function_call": {"name": "lookup", "args": {}}}]


def test_malformed_signature_dropped_not_raised():
    _, contents = _to_gemini_messages([
        _assistant_turn({
            "id": "t1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
            "thought_signature": "not!valid!base64",
        }),
    ])
    assert contents[0]["parts"] == [{"function_call": {"name": "lookup", "args": {}}}]


def test_decode_signature_round_trip():
    assert _decode_signature(base64.b64encode(SIG).decode("ascii")) == SIG
    assert _decode_signature(None) is None
    assert _decode_signature("") is None
    assert _decode_signature(b"already bytes") is None


def test_signature_survives_json_history_round_trip():
    """Memory backends json.dumps(model_dump()) — a bytes field would raise here."""
    original = ToolCall(id="t1", name="lookup", arguments={"q": "x"}, thought_signature=SIG_B64)
    message = Message(role="assistant", content="", tool_calls=[original])

    revived = Message(**json.loads(json.dumps(message.model_dump())))
    assert revived.tool_calls[0] == original

    _, contents = _to_gemini_messages([{
        "role": "assistant",
        "content": "",
        "tool_calls": [_tool_call_to_dict(tc) for tc in revived.tool_calls],
    }])
    assert contents[0]["parts"][0]["thought_signature"] == SIG


def test_old_history_row_without_signature_deserializes():
    """A Redis row written by an older SDK has no thought_signature key at all."""
    revived = Message(**{
        "role": "assistant",
        "content": "",
        "tool_call_id": None,
        "tool_calls": [{"id": "t1", "name": "lookup", "arguments": {}}],
    })
    assert revived.tool_calls[0].thought_signature is None


def test_tool_call_dict_omits_signature_for_non_gemini_providers():
    plain = ToolCall(id="t1", name="lookup", arguments={"a": 1})
    assert _tool_call_to_dict(plain) == {
        "id": "t1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"a": 1}'},
    }


def test_round_trip_against_real_genai_types():
    """Guards the assumption the fix rests on: thought_signature lives on Part.

    Skipped unless the gemini extra is installed. Still no network — this only
    builds and validates library objects.
    """
    types = pytest.importorskip("google.genai.types")

    assert "thought_signature" not in types.FunctionCall.model_fields
    assert types.Part.model_fields["thought_signature"].alias == "thoughtSignature"

    response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[
            types.Part(text="thinking"),
            types.Part(
                function_call=types.FunctionCall(id="t1", name="lookup", args={"q": "x"}),
                thought_signature=SIG,
            ),
        ]),
    )])

    calls = _tool_calls_from_response(response)
    assert [(c.id, c.name, c.thought_signature) for c in calls] == [("t1", "lookup", SIG_B64)]

    _, contents = _to_gemini_messages([
        _assistant_turn(_tool_call_to_dict(calls[0])),
    ])
    # the library must accept the part dict we hand back to it, signature intact
    part = types.Part.model_validate(contents[0]["parts"][0])
    assert part.function_call.name == "lookup"
    assert part.thought_signature == SIG


def test_openai_payload_unchanged_and_signature_stripped():
    plain = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "lookup", "arguments": "{}"}}]},
    ]
    assert _sanitize_messages(plain) == plain

    gemini_history = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "lookup", "arguments": "{}"},
                         "thought_signature": SIG_B64}]},
    ]
    assert _sanitize_messages(gemini_history) == plain[1:]
    # the caller's list is not mutated
    assert "thought_signature" in gemini_history[0]["tool_calls"][0]
