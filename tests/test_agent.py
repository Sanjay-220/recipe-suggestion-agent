"""Tests for the agent loop using a fake Anthropic client (no network calls)."""
from types import SimpleNamespace

import pytest

import recipe_agent.agent as agent_module
from recipe_agent.agent import latest_reply_text, run_turn


class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Message:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class _FakeRunner:
    """Mimics the `for message in runner` + generate_tool_call_response()
    contract described in the SDK's tool-use docs, without hitting the API.
    """

    def __init__(self, messages, tool_responses):
        self._messages = messages
        self._tool_responses = list(tool_responses)

    def __iter__(self):
        return iter(self._messages)

    def generate_tool_call_response(self):
        return self._tool_responses.pop(0) if self._tool_responses else None


def _fake_client(runners):
    """Stand-in for anthropic.Anthropic() whose tool_runner() yields the
    next pre-scripted _FakeRunner on each successive call."""
    runners_iter = iter(runners)
    beta = SimpleNamespace(
        messages=SimpleNamespace(tool_runner=lambda **kwargs: next(runners_iter))
    )
    return SimpleNamespace(beta=beta)


class TestRunTurn:
    def test_appends_final_assistant_reply_when_no_tool_call_is_made(self, monkeypatch):
        final_message = _Message([_TextBlock("Try the Vegetable Omelette!")], "end_turn")
        runner = _FakeRunner(messages=[final_message], tool_responses=[None])
        monkeypatch.setattr(agent_module, "_client", lambda: _fake_client([runner]))

        messages = [{"role": "user", "content": "I have eggs and cheese."}]
        result = run_turn(messages)

        assert result[-1] == {"role": "assistant", "content": final_message.content}

    def test_appends_tool_result_when_a_tool_is_called(self, monkeypatch):
        tool_call_message = _Message([_TextBlock("checking...")], "tool_use")
        final_message = _Message([_TextBlock("Here you go.")], "end_turn")
        tool_result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "[]"}]}
        runner = _FakeRunner(
            messages=[tool_call_message, final_message],
            tool_responses=[tool_result, None],
        )
        monkeypatch.setattr(agent_module, "_client", lambda: _fake_client([runner]))

        messages = [{"role": "user", "content": "What can I cook?"}]
        result = run_turn(messages)

        assert tool_result in result
        assert result[-1]["content"] == final_message.content

    def test_restarts_after_pause_turn_and_reaches_the_final_reply(self, monkeypatch):
        paused = _Message([_TextBlock("...")], "pause_turn")
        final_message = _Message([_TextBlock("Here's your recipe.")], "end_turn")
        runner_1 = _FakeRunner(messages=[paused], tool_responses=[None])
        runner_2 = _FakeRunner(messages=[final_message], tool_responses=[None])
        monkeypatch.setattr(
            agent_module, "_client", lambda: _fake_client([runner_1, runner_2])
        )

        result = run_turn([{"role": "user", "content": "What can I cook?"}])

        assert result[-1]["content"] == final_message.content

    def test_raises_runtime_error_after_max_pause_restarts(self, monkeypatch):
        paused = _Message([_TextBlock("...")], "pause_turn")
        runners = [_FakeRunner(messages=[paused], tool_responses=[None]) for _ in range(10)]
        monkeypatch.setattr(agent_module, "_client", lambda: _fake_client(runners))

        with pytest.raises(RuntimeError):
            run_turn([{"role": "user", "content": "hi"}])


class TestLatestReplyText:
    def test_extracts_text_from_the_last_assistant_message(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [_TextBlock("Hello there!")]},
        ]
        assert latest_reply_text(messages) == "Hello there!"

    def test_returns_empty_string_when_there_is_no_assistant_message(self):
        assert latest_reply_text([{"role": "user", "content": "hi"}]) == ""
