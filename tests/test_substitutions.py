"""Unit tests for ingredient substitution lookups."""
from recipe_agent.substitutions import suggest_substitution


class TestSuggestSubstitution:
    def test_returns_known_alternatives_for_a_common_ingredient(self):
        alternatives = suggest_substitution("butter")
        assert "margarine" in alternatives

    def test_is_case_and_whitespace_insensitive(self):
        assert suggest_substitution("  Butter  ") == suggest_substitution("butter")

    def test_returns_empty_list_for_unknown_ingredient(self):
        assert suggest_substitution("unobtainium") == []

    def test_normalizes_plural_and_qualifier_forms(self):
        assert suggest_substitution("Fresh Eggs") == suggest_substitution("egg")
