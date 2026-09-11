"""Ingredient-based filtering engine.

Pure Python, no network/LLM calls — this is the deterministic core that the
Claude agent calls as a tool. Kept dependency-free and separately testable so
match quality doesn't depend on model behavior.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "recipes.json"

# Descriptive words that don't affect whether an ingredient is "the same thing".
_QUALIFIERS = {
    "fresh", "chopped", "diced", "sliced", "minced", "ground",
    "large", "small", "medium", "ripe", "cooked", "raw", "dried",
}

# Plurals that a naive "strip trailing s" rule gets wrong.
_IRREGULAR_SINGULARS = {
    "tomatoes": "tomato",
    "potatoes": "potato",
    "leaves": "leaf",
    "loaves": "loaf",
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_ingredient(name: str) -> str:
    """Normalize an ingredient string for comparison.

    Lowercases, strips descriptive qualifiers ("fresh chopped tomatoes" ->
    "tomatoes"), collapses whitespace, and singularizes plurals so that
    "eggs" and "egg" (or "tomatoes" and "tomato") are treated as a match.
    """
    text = name.strip().lower()
    words = [w for w in _WHITESPACE_RE.split(text) if w and w not in _QUALIFIERS]
    words = [_singularize(w) for w in words]
    return " ".join(words)


def _singularize(word: str) -> str:
    if word in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[word]
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


@dataclass(frozen=True)
class MatchResult:
    """The outcome of scoring one recipe against a set of ingredients on hand."""

    recipe: dict
    coverage: float
    missing_ingredients: list[str]
    matched_count: int
    total_count: int


def load_recipes(path: str | Path | None = None) -> list[dict]:
    """Load the recipe dataset from a JSON file (defaults to data/recipes.json)."""
    target = Path(path) if path else DEFAULT_DATA_PATH
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def score_recipe(recipe: dict, user_ingredients: Iterable[str]) -> MatchResult:
    """Score a single recipe against the ingredients the user has on hand.

    Coverage is the fraction of the recipe's ingredients the user already
    has. Missing ingredients are returned in the recipe's original casing so
    they read naturally in a shopping-list style response.
    """
    normalized_user = {normalize_ingredient(i) for i in user_ingredients}

    matched = 0
    missing: list[str] = []
    for ingredient in recipe["ingredients"]:
        if normalize_ingredient(ingredient) in normalized_user:
            matched += 1
        else:
            missing.append(ingredient)

    total = len(recipe["ingredients"])
    coverage = matched / total if total else 0.0

    return MatchResult(
        recipe=recipe,
        coverage=coverage,
        missing_ingredients=missing,
        matched_count=matched,
        total_count=total,
    )


def filter_recipes(
    recipes: Iterable[dict],
    user_ingredients: Iterable[str],
    dietary_tags: Iterable[str] | None = None,
    cuisine: str | None = None,
    max_missing: int | None = None,
) -> list[MatchResult]:
    """Filter and rank recipes by how well they match ingredients on hand.

    Args:
        recipes: candidate recipes (as loaded by load_recipes()).
        user_ingredients: ingredients the user currently has.
        dietary_tags: if given, a recipe must carry every one of these tags
            (e.g. ["vegetarian", "gluten_free"]) to be included.
        cuisine: if given, only recipes with this exact cuisine are included.
        max_missing: if given, exclude recipes missing more than this many
            ingredients.

    Returns:
        MatchResult list sorted by fewest missing ingredients first, then by
        highest coverage, then alphabetically by recipe name (stable tie-break).
    """
    required_tags = set(dietary_tags) if dietary_tags else set()

    candidates = recipes
    if required_tags:
        candidates = (
            r for r in candidates if required_tags.issubset(set(r.get("dietary_tags", [])))
        )
    if cuisine:
        candidates = (r for r in candidates if r.get("cuisine") == cuisine)

    results = [score_recipe(r, user_ingredients) for r in candidates]

    if max_missing is not None:
        results = [r for r in results if len(r.missing_ingredients) <= max_missing]

    results.sort(
        key=lambda r: (len(r.missing_ingredients), -r.coverage, r.recipe["name"])
    )
    return results
