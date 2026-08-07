DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, bad anatomy, malformed hands, extra fingers, missing fingers, "
    "artifacts, text, watermark, blurry, duplicated body parts"
)

STYLE_TAGS = {
    "Anime": "anime style",
    "Detailed anime": "detailed anime illustration",
    "Cinematic anime": "cinematic anime scene, dramatic lighting",
    "Illustration": "polished digital illustration",
    "Manga-inspired": "manga-inspired illustration",
}


def enhance_prompt(prompt: str, style: str, quality_enhancer: bool) -> str:
    parts = [prompt.strip()]
    if style in STYLE_TAGS:
        parts.append(STYLE_TAGS[style])
    if quality_enhancer:
        parts.append("high detail, coherent composition, refined linework, professional lighting")
    return ", ".join(parts)
