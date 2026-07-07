"""Tests for the Gemini provider's OpenAI-format → Gemini contents conversion."""

from __future__ import annotations

from agentinc.sdk.providers.gemini import _to_gemini_messages


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
