from fizzy.prompts import AGENT_INSTRUCTIONS


class TestAgentInstructions:
    def test_is_string(self):
        assert isinstance(AGENT_INSTRUCTIONS, str)

    def test_not_empty(self):
        assert AGENT_INSTRUCTIONS.strip() != ""

    def test_contains_search_marker(self):
        assert "<<<<<<< SEARCH" in AGENT_INSTRUCTIONS

    def test_contains_separator_marker(self):
        assert "=======" in AGENT_INSTRUCTIONS

    def test_contains_replace_marker(self):
        assert ">>>>>>> REPLACE" in AGENT_INSTRUCTIONS

    def test_mentions_uniqueness(self):
        text = AGENT_INSTRUCTIONS.lower()
        assert "once" in text or "unique" in text
