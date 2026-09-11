"""Unit tests for the ingredient-based filtering engine."""
from recipe_agent.filtering import (
    normalize_ingredient,
    score_recipe,
    filter_recipes,
    load_recipes,
)

SAMPLE_RECIPES = [
    {
        "id": "r1",
        "name": "Simple Omelette",
        "cuisine": "american",
        "dietary_tags": ["vegetarian", "gluten_free"],
        "ingredients": ["egg", "cheese", "butter"],
        "instructions": ["Cook eggs with cheese in butter."],
        "prep_time_minutes": 10,
    },
    {
        "id": "r2",
        "name": "Beef Tacos",
        "cuisine": "mexican",
        "dietary_tags": [],
        "ingredients": ["ground beef", "taco shell", "cheese", "tomato"],
        "instructions": ["Cook beef, fill shells."],
        "prep_time_minutes": 20,
    },
    {
        "id": "r3",
        "name": "Vegan Chickpea Curry",
        "cuisine": "indian",
        "dietary_tags": ["vegetarian", "vegan", "dairy_free", "gluten_free"],
        "ingredients": ["chickpea", "tomato", "onion", "garlic", "cumin"],
        "instructions": ["Simmer chickpeas in spiced tomato sauce."],
        "prep_time_minutes": 30,
    },
]


class TestNormalizeIngredient:
    def test_lowercases_and_strips_whitespace(self):
        assert normalize_ingredient("  Egg  ") == "egg"

    def test_strips_common_qualifiers(self):
        assert normalize_ingredient("Fresh Chopped Tomatoes") == "tomato"

    def test_handles_regular_plurals(self):
        assert normalize_ingredient("onions") == "onion"

    def test_handles_irregular_plurals(self):
        assert normalize_ingredient("tomatoes") == "tomato"
        assert normalize_ingredient("potatoes") == "potato"


class TestScoreRecipe:
    def test_full_match_has_coverage_one_and_no_missing(self):
        recipe = SAMPLE_RECIPES[0]
        result = score_recipe(recipe, {"egg", "cheese", "butter", "salt"})
        assert result.coverage == 1.0
        assert result.missing_ingredients == []
        assert result.matched_count == 3
        assert result.total_count == 3

    def test_partial_match_lists_missing_ingredients(self):
        recipe = SAMPLE_RECIPES[0]
        result = score_recipe(recipe, {"egg", "cheese"})
        assert result.matched_count == 2
        assert result.missing_ingredients == ["butter"]
        assert result.coverage == 2 / 3

    def test_no_overlap_has_zero_coverage(self):
        recipe = SAMPLE_RECIPES[0]
        result = score_recipe(recipe, {"spaghetti", "olive oil"})
        assert result.coverage == 0.0
        assert result.matched_count == 0
        assert set(result.missing_ingredients) == {"egg", "cheese", "butter"}

    def test_matching_is_normalization_aware(self):
        recipe = SAMPLE_RECIPES[0]
        # user has plural + qualifier forms; should still match
        result = score_recipe(recipe, {"Eggs", "Fresh Cheese", "butter"})
        assert result.coverage == 1.0


class TestFilterRecipes:
    def test_ranks_fewest_missing_first(self):
        user_ingredients = {"egg", "cheese", "butter", "tomato"}
        ranked = filter_recipes(SAMPLE_RECIPES, user_ingredients)
        # r1 (0 missing) should outrank r2 (2 missing) and r3 (3 missing)
        assert ranked[0].recipe["id"] == "r1"

    def test_excludes_recipes_missing_a_required_dietary_tag(self):
        user_ingredients = {"egg", "cheese", "butter", "ground beef", "taco shell", "tomato"}
        ranked = filter_recipes(
            SAMPLE_RECIPES, user_ingredients, dietary_tags=["vegetarian"]
        )
        ids = {m.recipe["id"] for m in ranked}
        assert "r2" not in ids  # beef tacos has no dietary tags
        assert "r1" in ids

    def test_filters_by_cuisine(self):
        user_ingredients = {"chickpea", "tomato", "onion", "garlic", "cumin"}
        ranked = filter_recipes(SAMPLE_RECIPES, user_ingredients, cuisine="indian")
        assert {m.recipe["id"] for m in ranked} == {"r3"}

    def test_respects_max_missing(self):
        user_ingredients = {"egg"}
        ranked = filter_recipes(SAMPLE_RECIPES, user_ingredients, max_missing=1)
        # r1 is missing 2 (cheese, butter) -> excluded when max_missing=1
        assert all(m.recipe["id"] != "r1" for m in ranked)

    def test_returns_empty_list_when_no_recipe_carries_the_required_tag(self):
        ranked = filter_recipes(
            SAMPLE_RECIPES, {"egg"}, dietary_tags=["keto"]
        )
        assert ranked == []


class TestLoadRecipes:
    def test_loads_default_dataset_as_list_of_dicts(self):
        recipes = load_recipes()
        assert isinstance(recipes, list)
        assert len(recipes) > 0
        assert all("ingredients" in r for r in recipes)
