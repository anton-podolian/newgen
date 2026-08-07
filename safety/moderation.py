from __future__ import annotations

import re

MAX_PROMPT_LENGTH = 2000

_MINOR_TERMS = re.compile(
    r"\b(minors?|under[ -]?age|children?|child|loli(?:con)?|shota(?:con)?|school[ -]?child|"
    r"schoolgirl|schoolboy|toddler|pre[ -]?teen|teenager|young-looking|little girl|little boy|"
    r"middle school(?:er)?|elementary school(?:er)?|high school(?:er)?)\b", re.IGNORECASE
)
_UNDER_18 = re.compile(
    r"\b(?:age[ds]?\s*)?(?:[0-9]|1[0-7])\s*(?:years? old|y/?o|yo|yrs?)\b|"
    r"\b(?:[0-9]|1[0-7])[- ]year[- ]old\b", re.IGNORECASE
)
_SEXUAL = re.compile(
    r"\b(nsfw|nude|naked|sex(?:ual|ually)?|explicit|erotic|porn(?:ographic)?|intercourse|"
    r"genitals?|breasts?|nipples?|vagina|penis|fetish|seductive)\b", re.IGNORECASE
)
_REAL_PERSON = re.compile(
    r"\b(real (?:person|woman|man|girl|boy|celebrity|actor|actress|singer)|celebrity|"
    r"deep[ -]?fake|photo of|photograph of|looks? (?:exactly )?like|face swap)\b", re.IGNORECASE
)


def validate_prompt(prompt: str) -> tuple[bool, str]:
    """Conservative MVP filter; designed to be replaceable with a moderation API."""
    text = " ".join((prompt or "").split())
    if not text:
        return False, "Prompt cannot be empty."
    if len(text) > MAX_PROMPT_LENGTH:
        return False, f"Prompt is too long (maximum {MAX_PROMPT_LENGTH} characters)."
    if _MINOR_TERMS.search(text) or _UNDER_18.search(text):
        return False, "Requests involving minors or young/ambiguous-age characters are not allowed."
    if _SEXUAL.search(text) and _REAL_PERSON.search(text):
        return False, "Sexual content involving real people or impersonation is not allowed."
    return True, "OK"
