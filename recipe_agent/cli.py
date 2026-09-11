"""Command-line interface: one-shot and interactive REPL modes."""
from __future__ import annotations

import argparse
import os
import sys

import anthropic

from recipe_agent.agent import latest_reply_text, run_turn

VALID_DIETARY_TAGS = {"vegetarian", "vegan", "gluten_free", "dairy_free"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recipe-agent",
        description="Suggest recipes based on ingredients you have on hand.",
    )
    parser.add_argument(
        "--ingredients",
        "-i",
        help='Comma-separated ingredients, e.g. "eggs, spinach, feta cheese". '
        "Omit to start an interactive session instead.",
    )
    parser.add_argument(
        "--diet",
        "-d",
        action="append",
        dest="dietary_tags",
        choices=sorted(VALID_DIETARY_TAGS),
        help="Dietary restriction to require. Repeat for multiple, e.g. -d vegetarian -d gluten_free.",
    )
    parser.add_argument("--cuisine", "-c", help='Cuisine preference, e.g. "italian".')
    return parser


def _build_user_message(ingredients: str, dietary_tags: list[str] | None, cuisine: str | None) -> str:
    parts = [f"I have these ingredients: {ingredients}."]
    if dietary_tags:
        parts.append(f"Dietary requirements: {', '.join(dietary_tags)}.")
    if cuisine:
        parts.append(f"I'd prefer {cuisine} food.")
    parts.append("What can I make?")
    return " ".join(parts)


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "Authentication failed. Check that ANTHROPIC_API_KEY is set correctly "
            "(see .env.example)."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "Your API key doesn't have permission for this request."
    if isinstance(exc, anthropic.NotFoundError):
        return "The requested model or endpoint wasn't found. Check CLAUDE_MODEL in .env."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the API. Wait a moment and try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check your network connection."
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error ({exc.status_code}): {exc.message}"
    return f"Unexpected error: {exc}"


def run_one_shot(ingredients: str, dietary_tags: list[str] | None, cuisine: str | None) -> int:
    if not ingredients.strip():
        print("Error: --ingredients cannot be empty.", file=sys.stderr)
        return 1

    user_message = _build_user_message(ingredients, dietary_tags, cuisine)
    messages = [{"role": "user", "content": user_message}]

    try:
        messages = run_turn(messages)
    except Exception as exc:  # noqa: BLE001 - translated to a friendly message below
        print(_friendly_error(exc), file=sys.stderr)
        return 1

    print(latest_reply_text(messages))
    return 0


def run_interactive() -> int:
    print("Recipe Suggestion Agent — tell me what you have. Type 'quit' to exit.\n")
    messages: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        messages.append({"role": "user", "content": user_input})
        try:
            messages = run_turn(messages)
        except Exception as exc:  # noqa: BLE001
            print(_friendly_error(exc), file=sys.stderr)
            continue

        print(f"\nAgent: {latest_reply_text(messages)}\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Warning: ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
            "and add your key.",
            file=sys.stderr,
        )

    if args.ingredients:
        return run_one_shot(args.ingredients, args.dietary_tags, args.cuisine)
    return run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
