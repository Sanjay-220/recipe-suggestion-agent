"""The Claude-powered agent loop.

Wires the deterministic tools in recipe_agent.tools to an Anthropic Tool
Runner session. The model decides *when* to call the filtering tool and how
to phrase the answer; it never invents recipes outside what the tool
returns (enforced via the system prompt).
"""
from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from recipe_agent.tools import ALL_TOOLS

load_dotenv()

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 4096
MAX_PAUSE_RESTARTS = 3

SYSTEM_PROMPT = """You are a friendly, practical recipe suggestion assistant.

The user tells you what ingredients they have on hand, and may mention
dietary restrictions (vegetarian, vegan, gluten_free, dairy_free) or a
cuisine preference. Follow this process:

1. ALWAYS call search_recipes_by_ingredients before recommending anything.
   Never suggest a recipe from general knowledge — only from what the tool
   returns. Pass dietary_tags/cuisine through if the user mentioned them.
2. Present the top 2-4 matches conversationally: the recipe name, why it's
   a good match ("you already have everything you need" or "just missing
   one item"), and its missing ingredients if any.
3. For any missing ingredient, call suggest_substitution and mention a
   substitute if one exists, so the user isn't blocked on a grocery run.
4. If the user asks for full instructions on a specific recipe, call
   get_recipe_details and walk them through the steps.
5. Be concise and warm. Translate tool results into natural language —
   never dump raw JSON at the user.
"""


def _client() -> anthropic.Anthropic:
    # anthropic.Anthropic() reads ANTHROPIC_API_KEY from the environment.
    return anthropic.Anthropic()


def _model() -> str:
    return os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)


def run_turn(messages: list[dict]) -> list[dict]:
    """Run one full agent turn — including any tool calls — in place.

    `messages` must already end with the new user turn. Mutates and returns
    the same list with the assistant's tool calls, tool results, and final
    reply appended, so the caller can pass it straight back in on the next
    user turn for multi-turn memory.
    """
    client = _client()
    model = _model()

    restarts = 0
    while True:
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=messages,
        )

        last = None
        for message in runner:
            last = message
            # Mirror history ourselves — the runner keeps its own internal
            # copy and doesn't expose it, but we need it to resume after a
            # pause_turn and to preserve multi-turn memory across calls.
            messages.append({"role": "assistant", "content": message.content})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                messages.append(tool_response)

        if last is None or last.stop_reason != "pause_turn":
            break
        restarts += 1
        if restarts > MAX_PAUSE_RESTARTS:
            raise RuntimeError(
                "Agent turn stayed paused after the maximum number of restarts."
            )

    return messages


def latest_reply_text(messages: list[dict]) -> str:
    """Extract the plain text of the most recent assistant message."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        blocks = message.get("content") or []
        texts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
        if texts:
            return "\n".join(texts)
    return ""
