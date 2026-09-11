# Recipe Suggestion Agent

An AI agent that suggests recipes based on ingredients you already have.
It pairs **Claude (Anthropic API, tool calling)** with a deterministic,
unit-tested **ingredient-based filtering engine** — the LLM handles
understanding the user and explaining results conversationally; a plain
Python module handles the actual matching, so recommendations are grounded
in real data instead of the model's general knowledge.

## How it works

```
 User (CLI)
     |
     v
 Claude (claude-opus-5, tool calling)  <-- system prompt: always filter before answering
     |
     |  calls tool
     v
 recipe_agent/tools.py                 (thin JSON adapters)
     |
     v
 recipe_agent/filtering.py             (pure Python — ingredient matching,
     |                                  dietary/cuisine filters, ranking)
     v
 data/recipes.json                     (50 recipes, local, no external API)
```

1. The user describes what ingredients they have (and optionally dietary
   restrictions or a cuisine preference), via CLI flags or an interactive
   chat.
2. Claude is instructed to **always** call `search_recipes_by_ingredients`
   before recommending anything — it never invents a recipe from general
   knowledge.
3. The filtering engine (`recipe_agent/filtering.py`) normalizes ingredient
   names (case, plurals, descriptive qualifiers like "fresh chopped"),
   scores every candidate recipe by ingredient coverage, applies any
   dietary-tag / cuisine filters, and ranks by fewest missing ingredients.
4. For any missing ingredient, Claude can call `suggest_substitution` to
   offer a common swap (e.g. butter → margarine) instead of leaving the
   user stuck.
5. Claude turns the ranked, structured results into a natural-language
   answer.

This hybrid split — deterministic filtering + LLM explanation — means the
matching logic is fast, free, fully testable, and doesn't depend on model
behavior for correctness; the LLM adds flexible natural-language input and
friendly, contextual explanations on top.

## Setup

```bash
# 1. Clone / cd into the project
cd recipe-suggestion-agent

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at https://console.anthropic.com/settings/keys.

By default the agent uses `claude-opus-5`. For a lower-cost run (e.g. for
repeated assignment testing), set `CLAUDE_MODEL=claude-sonnet-5` or
`CLAUDE_MODEL=claude-haiku-4-5` in `.env`.

## Usage

**One-shot mode:**

```bash
python main.py --ingredients "eggs, spinach, feta cheese" --diet vegetarian
python main.py -i "ground beef, taco shell, tomato, cheese" --cuisine mexican
```

**Interactive mode** (holds conversation memory across turns):

```bash
python main.py
```

```
Recipe Suggestion Agent — tell me what you have. Type 'quit' to exit.

You: I have eggs, spinach, and feta cheese
Agent: Spinach Feta Scramble is a great fit — you already have everything
you need! ...

You: what if I also have chicken?
Agent: ...
```

See `docs/example_run.md` for a full sample transcript.

## Project layout

```
data/recipes.json          Local recipe dataset (50 recipes: american, italian,
                           indian, south_indian, mexican, chinese, thai,
                           japanese, mediterranean)
recipe_agent/filtering.py  Ingredient normalization, scoring, ranking (pure Python)
recipe_agent/substitutions.py  Static ingredient-substitution lookup
recipe_agent/tools.py      @beta_tool-wrapped adapters Claude calls
recipe_agent/agent.py      Anthropic Tool Runner loop + system prompt
recipe_agent/cli.py        One-shot and interactive CLI
main.py                    Entry point
tests/                     pytest unit tests (filtering, substitutions, tools, agent, cli)
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=recipe_agent --cov-report=term-missing
```

91% overall coverage; the filtering engine and tool layer (the graded core
logic) are at 98–100%. The agent/CLI network paths are covered with a
scripted fake Anthropic client (see `tests/test_agent.py`) rather than live
API calls, so the suite runs offline and deterministically.

## Design notes / known limitations

- **Ingredient matching** is name-based (normalized string matching), not
  semantic — it won't know "chicken breast" and "chicken thigh" are both
  "chicken" unless written that way in the dataset, and it has no
  quantity/unit awareness (it only checks whether an ingredient is present).
- **Dataset is static and local** (50 recipes) by design, per project scope
  — no external recipe API dependency, so the agent runs fully offline
  except for the Claude call itself.
- **Substitutions** come from a small hardcoded table, not a live lookup —
  easy to extend in `recipe_agent/substitutions.py`.
- The Python SDK's Tool Runner is beta; `agent.py` includes the documented
  restart-on-`pause_turn` workaround since the runner doesn't auto-resume a
  paused turn.
