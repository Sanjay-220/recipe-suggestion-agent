"""Tests for CLI argument parsing and input validation (no network calls)."""
import anthropic
import httpx
import pytest

from recipe_agent.cli import (
    _build_user_message,
    _friendly_error,
    build_parser,
    run_interactive,
    run_one_shot,
)


class TestBuildParser:
    def test_parses_ingredients_flag(self):
        args = build_parser().parse_args(["--ingredients", "egg, spinach"])
        assert args.ingredients == "egg, spinach"

    def test_diet_flag_can_repeat(self):
        args = build_parser().parse_args(["-d", "vegetarian", "-d", "gluten_free"])
        assert args.dietary_tags == ["vegetarian", "gluten_free"]

    def test_rejects_unknown_dietary_tag(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["-d", "keto"])

    def test_ingredients_optional_for_interactive_mode(self):
        args = build_parser().parse_args([])
        assert args.ingredients is None


class TestBuildUserMessage:
    def test_includes_ingredients(self):
        message = _build_user_message("egg, spinach", None, None)
        assert "egg, spinach" in message

    def test_includes_dietary_tags_when_given(self):
        message = _build_user_message("egg", ["vegetarian"], None)
        assert "vegetarian" in message

    def test_includes_cuisine_when_given(self):
        message = _build_user_message("egg", None, "italian")
        assert "italian" in message


class TestFriendlyError:
    def test_authentication_error_mentions_api_key(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(401, request=request)
        exc = anthropic.AuthenticationError(message="bad key", response=response, body=None)
        assert "ANTHROPIC_API_KEY" in _friendly_error(exc)

    def test_unknown_exception_falls_back_to_generic_message(self):
        assert "Unexpected error" in _friendly_error(ValueError("boom"))

    def test_rate_limit_error_mentions_waiting(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(429, request=request)
        exc = anthropic.RateLimitError(message="slow down", response=response, body=None)
        assert "Wait a moment" in _friendly_error(exc)

    def test_not_found_error_mentions_model(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(404, request=request)
        exc = anthropic.NotFoundError(message="no such model", response=response, body=None)
        assert "CLAUDE_MODEL" in _friendly_error(exc)

    def test_connection_error_mentions_network(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.APIConnectionError(request=request)
        assert "network" in _friendly_error(exc).lower()


class TestRunOneShot:
    def test_rejects_empty_ingredients_without_calling_the_api(self, capsys):
        exit_code = run_one_shot("   ", None, None)
        assert exit_code == 1
        assert "cannot be empty" in capsys.readouterr().err

    def test_prints_final_reply_and_returns_zero_on_success(self, monkeypatch, capsys):
        monkeypatch.setattr("recipe_agent.cli.run_turn", lambda messages: messages)
        monkeypatch.setattr("recipe_agent.cli.latest_reply_text", lambda messages: "Try the tacos!")

        exit_code = run_one_shot("beef, tortilla", None, None)

        assert exit_code == 0
        assert "Try the tacos!" in capsys.readouterr().out

    def test_prints_friendly_message_and_returns_one_on_failure(self, monkeypatch, capsys):
        def failing_run_turn(messages):
            raise ValueError("boom")

        monkeypatch.setattr("recipe_agent.cli.run_turn", failing_run_turn)

        exit_code = run_one_shot("beef, tortilla", None, None)

        assert exit_code == 1
        assert "Unexpected error" in capsys.readouterr().err


class TestRunInteractive:
    def test_exits_immediately_on_quit_without_calling_run_turn(self, monkeypatch):
        inputs = iter(["quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        def unexpected_call(messages):
            raise AssertionError("run_turn should not be called before any real input")

        monkeypatch.setattr("recipe_agent.cli.run_turn", unexpected_call)

        assert run_interactive() == 0

    def test_processes_one_turn_then_quits(self, monkeypatch, capsys):
        inputs = iter(["I have eggs", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        monkeypatch.setattr("recipe_agent.cli.run_turn", lambda messages: messages)
        monkeypatch.setattr("recipe_agent.cli.latest_reply_text", lambda messages: "Scramble them!")

        exit_code = run_interactive()

        assert exit_code == 0
        assert "Scramble them!" in capsys.readouterr().out

    def test_ignores_blank_input_lines(self, monkeypatch, capsys):
        inputs = iter(["", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        def unexpected_call(messages):
            raise AssertionError("run_turn should not be called for blank input")

        monkeypatch.setattr("recipe_agent.cli.run_turn", unexpected_call)

        assert run_interactive() == 0
