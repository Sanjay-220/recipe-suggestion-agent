"""Static ingredient-substitution lookup.

Deliberately simple and deterministic — a hardcoded table is easy to audit,
easy to extend, and doesn't need an LLM call for something this well-known.
"""
from __future__ import annotations

from recipe_agent.filtering import normalize_ingredient

SUBSTITUTIONS: dict[str, list[str]] = {
    "butter": ["margarine", "olive oil", "coconut oil"],
    "milk": ["almond milk", "soy milk", "oat milk"],
    "egg": ["flax egg", "chia egg", "applesauce (for baking)"],
    "cream": ["coconut cream", "evaporated milk"],
    "heavy cream": ["coconut cream", "evaporated milk"],
    "sour cream": ["greek yogurt"],
    "yogurt": ["sour cream", "coconut yogurt"],
    "buttermilk": ["milk + a splash of lemon juice"],
    "white wine": ["chicken broth", "diluted apple cider vinegar"],
    "parmesan": ["nutritional yeast", "pecorino"],
    "soy sauce": ["tamari", "coconut aminos"],
    "fish sauce": ["soy sauce + a squeeze of lime"],
    "mirin": ["rice vinegar + a pinch of sugar"],
    "tahini": ["peanut butter", "sunflower seed butter"],
    "feta cheese": ["goat cheese"],
    "ground beef": ["ground turkey", "lentils"],
    "chicken": ["tofu", "chickpeas"],
    "beef": ["mushroom", "tofu"],
    "cheese": ["nutritional yeast", "cashew cheese"],
}


def suggest_substitution(ingredient: str) -> list[str]:
    """Return known substitutes for a missing ingredient, or [] if unknown."""
    return SUBSTITUTIONS.get(normalize_ingredient(ingredient), [])
