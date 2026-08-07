from utils.prompts import enhance_prompt


def test_user_prompt_stays_first():
    result = enhance_prompt("red-haired adult mage", "Anime", True)
    assert result.startswith("red-haired adult mage")
    assert "anime style" in result
