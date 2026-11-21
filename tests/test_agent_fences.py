"""Fenced tool-call parsing edge cases (app/agent.py _parse_fenced_calls).

A model can fence a ```tool_call block whose JSON parses fine but is not
an object — a list, a bare string, a number. Before the fix that crashed
the whole turn with AttributeError on .get; the block must be skipped
like any other malformed block, and the surrounding text kept.
"""

from app.agent import _parse_fenced_calls


def test_fenced_call_with_json_list_is_skipped_not_crash():
    clean, calls = _parse_fenced_calls(
        'hi ```tool_call\n[{"name": "shell_exec"}]\n```')
    assert calls == []
    assert clean == "hi"


def test_fenced_call_with_json_string_is_skipped_not_crash():
    clean, calls = _parse_fenced_calls(
        'hi ```tool_call\n"just a string"\n```')
    assert calls == []
    assert clean == "hi"


def test_fenced_call_with_json_number_is_skipped_not_crash():
    clean, calls = _parse_fenced_calls("hi ```tool_call\n42\n```")
    assert calls == []
    assert clean == "hi"


def test_fenced_call_with_json_null_is_skipped_not_crash():
    clean, calls = _parse_fenced_calls("hi ```tool_call\nnull\n```")
    assert calls == []
    assert clean == "hi"


def test_non_object_block_does_not_block_a_good_call_after_it():
    clean, calls = _parse_fenced_calls(
        '```tool_call\n["nope"]\n```\ntext\n'
        '```tool_call\n{"name": "file_list", "arguments": {"path": "."}}\n```')
    assert len(calls) == 1
    assert calls[0]["name"] == "file_list"
    assert calls[0]["args"] == {"path": "."}
    assert clean == "text"


def test_object_without_name_is_still_skipped():
    clean, calls = _parse_fenced_calls(
        '```tool_call\n{"arguments": {"path": "."}}\n```')
    assert calls == []
    assert clean == ""
