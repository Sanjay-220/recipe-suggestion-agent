"""Claude-callable tools wrapping the deterministic filtering engine.

Each function below is decorated with @beta_tool so the Anthropic SDK's tool
runner can generate its schema from the signature + docstring and call it
directly. The functions themselves do no LLM reasoning — they're thin,
JSON-returning adapters over recipe_agent.filtering / recipe_agent.substitutions,
which keeps the actual matching logic unit-testable in isolation (see
tests/test_filtering.py).
"""
from __future__ import annotations

import json

from anthropic import beta_tool

from recipe_agent.filtering import MatchResult, filter_recipes, load_recipes
from recipe_agent.substitutions import suggest_substitution as _suggest_substitution

_RECIPES = load_recipes()


def _match_to_dict(match: MatchResult) -> dict:
    return {
        "id": match.recipe["id"],
        "name": match.recipe["name"],
        "cuisine": match.recipe.get("cuisine"),
        "dietary_tags": match.recipe.get("dietary_tags", []),
        "prep_time_minutes": match.recipe.get("prep_time_minutes"),
        "coverage_percent": round(match.coverage * 100),
        "matched_count": match.matched_count,
        "total_count": match.total_count,
        "missing_ingredients": match.missing_ingredients,
    }


@beta_tool
def search_recipes_by_ingredients(
    ingredients: list[str],
    dietary_tags: list[str] | None = None,
    cuisine: str | None = None,
    max_missing: int | None = None,
    max_results: int = 5,
) -> str:
    """Find recipes ranked by how well they match ingredients the user has on hand.

    Always call this before suggesting any recipe by name — it is the source
    of truth for what's actually makeable, not general knowledge.

    Args:
        ingredients: Ingredients the user currently has, e.g. ["egg", "spinach", "feta cheese"].
        dietary_tags: Optional required dietary tags, e.g. ["vegetarian", "gluten_free"]. A recipe must satisfy every given tag to be included.
        cuisine: Optional cuisine filter, e.g. "italian". Must exactly match the recipe's cuisine.
        max_missing: Optional cap on how many ingredients a recipe may be missing to still be included.
        max_results: Maximum number of ranked recipes to return.
    """
    results = filter_recipes(
        _RECIPES,
        ingredients,
        dietary_tags=dietary_tags,
        cuisine=cuisine,
        max_missing=max_missing,
    )
    top = results[:max_results]
    return json.dumps(
        {
            "total_matches": len(results),
            "returned": len(top),
            "recipes": [_match_to_dict(m) for m in top],
        },
        indent=2,
    )


@beta_tool
def get_recipe_details(recipe_id: str) -> str:
    """Get full details for one recipe: all ingredients and step-by-step instructions.

    Args:
        recipe_id: The recipe's id, as returned by search_recipes_by_ingredients (e.g. "veg-omelette").
    """
    for recipe in _RECIPES:
        if recipe["id"] == recipe_id:
            return json.dumps(recipe, indent=2)
    return json.dumps({"error": f"No recipe found with id '{recipe_id}'."})


@beta_tool
def suggest_substitution(ingredient: str) -> str:
    """Suggest common substitutes for one missing ingredient.

    Args:
        ingredient: The ingredient to find a substitute for, e.g. "buttermilk".
    """
    alternatives = _suggest_substitution(ingredient)
    return json.dumps({"ingredient": ingredient, "alternatives": alternatives})


ALL_TOOLS = [search_recipes_by_ingredients, get_recipe_details, suggest_substitution]
