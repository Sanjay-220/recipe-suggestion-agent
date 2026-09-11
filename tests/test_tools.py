"""Unit tests for the @beta_tool-wrapped functions Claude calls.

BetaFunctionTool instances remain directly callable with keyword args, so
these exercise the same code the tool runner would invoke, without needing
a live API call.
"""
import json

from recipe_agent.tools import (
    get_recipe_details,
    search_recipes_by_ingredients,
    suggest_substitution,
)


class TestSearchRecipesByIngredients:
    def test_returns_json_with_ranked_recipes(self):
        raw = search_recipes_by_ingredients(ingredients=["egg", "cheese", "butter"])
        data = json.loads(raw)
        assert data["returned"] > 0
        assert all("missing_ingredients" in r for r in data["recipes"])

    def test_respects_max_results(self):
        raw = search_recipes_by_ingredients(ingredients=["egg"], max_results=2)
        data = json.loads(raw)
        assert data["returned"] <= 2

    def test_applies_dietary_tags_filter(self):
        raw = search_recipes_by_ingredients(
            ingredients=["ground beef", "taco shell"], dietary_tags=["vegan"]
        )
        data = json.loads(raw)
        assert all("vegan" in r.get("dietary_tags", []) for r in data["recipes"])

    def test_applies_cuisine_filter(self):
        raw = search_recipes_by_ingredients(ingredients=["chickpea"], cuisine="indian")
        data = json.loads(raw)
        assert all(r["cuisine"] == "indian" for r in data["recipes"])


class TestGetRecipeDetails:
    def test_returns_full_recipe_for_known_id(self):
        raw = get_recipe_details(recipe_id="veg-omelette")
        data = json.loads(raw)
        assert data["name"] == "Vegetable Omelette"
        assert "instructions" in data

    def test_returns_error_payload_for_unknown_id(self):
        raw = get_recipe_details(recipe_id="not-a-real-recipe")
        data = json.loads(raw)
        assert "error" in data


class TestSuggestSubstitutionTool:
    def test_returns_alternatives_for_known_ingredient(self):
        raw = suggest_substitution(ingredient="butter")
        data = json.loads(raw)
        assert "margarine" in data["alternatives"]

    def test_returns_empty_alternatives_for_unknown_ingredient(self):
        raw = suggest_substitution(ingredient="unobtainium")
        data = json.loads(raw)
        assert data["alternatives"] == []
