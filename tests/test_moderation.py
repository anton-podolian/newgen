from safety.moderation import validate_prompt


def test_empty_rejected():
    assert not validate_prompt("  ")[0]


def test_minor_terms_and_ages_rejected():
    for prompt in ("a child", "17 years old", "16-year-old", "loli", "young-looking woman"):
        assert not validate_prompt(prompt)[0], prompt


def test_adult_fictional_nsfw_not_blanket_blocked():
    assert validate_prompt("fictional 25-year-old adult woman, erotic anime illustration")[0]


def test_real_person_sexual_request_rejected():
    assert not validate_prompt("explicit nude photo of a real celebrity")[0]
