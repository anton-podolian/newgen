from __future__ import annotations

import atexit
import logging
import os
from functools import lru_cache

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

from generator import HuggingFaceImageGenerator, ProviderError
from utils import DEFAULT_NEGATIVE_PROMPT, enhance_prompt

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# This is the fal provider ID from the live Hub mapping for SDXL 1.0.
# Hugging Face's fal router routes by provider ID, not by `/models/<Hub model>`.
DEFAULT_API_URL = "https://router.huggingface.co/fal-ai/fal-ai/fast-sdxl"
DEFAULT_LORA_REPOSITORY = "serenitymea/Myanime_model"
DEFAULT_LORA_WEIGHT_NAME = "Anime.safetensors"

MIN_DIMENSION = 512
MAX_DIMENSION = 1536
DIMENSION_STEP = 64
MIN_STEPS = 10
MAX_STEPS = 50
MIN_IMAGES = 1
MAX_IMAGES = 4

ASPECT_SIZES = {
    "Portrait": (832, 1216),
    "Square": (1024, 1024),
    "Landscape": (1216, 832),
    "Custom": None,
}
ALLOWED_DIMENSIONS = range(MIN_DIMENSION, MAX_DIMENSION + 1, DIMENSION_STEP)


@lru_cache(maxsize=1)
def get_generator() -> HuggingFaceImageGenerator:
    return HuggingFaceImageGenerator(
        api_key=os.getenv("HF_TOKEN", ""),
        model=os.getenv("IMAGE_MODEL", DEFAULT_MODEL),
        base_url=os.getenv("IMAGE_API_URL", DEFAULT_API_URL),
        timeout=float(os.getenv("IMAGE_API_TIMEOUT", "120")),
    )


def close_generator() -> None:
    if get_generator.cache_info().currsize:
        get_generator().close()


atexit.register(close_generator)


def set_aspect_size(aspect: str) -> tuple[gr.Slider, gr.Slider]:
    size = ASPECT_SIZES.get(aspect)
    if size is None:
        return gr.Slider(interactive=True), gr.Slider(interactive=True)
    return gr.Slider(value=size[0], interactive=False), gr.Slider(value=size[1], interactive=False)


def validate_generation(width: int, height: int, steps: int, image_count: int) -> None:
    if width not in ALLOWED_DIMENSIONS or height not in ALLOWED_DIMENSIONS:
        raise gr.Error("Width and height must be 512 to 1536 pixels in steps of 64.")
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise gr.Error("Steps must be between 10 and 50.")
    if not MIN_IMAGES <= image_count <= MAX_IMAGES:
        raise gr.Error("Choose between 1 and 4 images.")


def generate_images(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    random_seed: bool,
    image_count: int,
    style: str,
    add_quality_tags: bool,
    use_lora: bool,
    lora_strength: float,
) -> tuple[list[tuple[Image.Image, str]], str]:
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Enter a prompt.")

    width, height = int(width), int(height)
    steps, image_count, seed = int(steps), int(image_count), int(seed)
    validate_generation(width, height, steps, image_count)

    final_prompt = enhance_prompt(prompt, style, add_quality_tags)
    images: list[tuple[Image.Image, str]] = []
    seeds: list[int] = []
    try:
        generator = get_generator()
        for index in range(image_count):
            image_seed = None if random_seed or seed < 0 else seed + index
            generated = generator.generate(
                prompt=final_prompt,
                negative_prompt=negative_prompt.strip(),
                width=width,
                height=height,
                steps=steps,
                guidance_scale=float(guidance),
                seed=image_seed,
                lora_repository=DEFAULT_LORA_REPOSITORY if use_lora else None,
                lora_weight_name=os.getenv("LORA_WEIGHT_NAME") or DEFAULT_LORA_WEIGHT_NAME,
                lora_scale=float(lora_strength),
            )
            images.append((generated.image, f"Seed {generated.seed}"))
            seeds.append(generated.seed)
    except (ProviderError, ValueError) as exc:
        logger.exception("Generation failed")
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected generation failure")
        raise gr.Error("Unexpected generation error. Please try again.") from exc

    return images, generation_details(image_count, seeds, width, height, steps, guidance, final_prompt)


def generation_details(
    image_count: int,
    seeds: list[int],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    prompt: str,
) -> str:
    return (
        f"**Generated {image_count} image(s)**  \n"
        f"**Seeds:** {', '.join(map(str, seeds))}  \n"
        f"**Parameters:** {width} x {height}, {steps} steps, CFG {guidance:g}  \n"
        f"**Final prompt:** {prompt}"
    )


CSS = """
.container {
    max-width: 1120px !important;
    margin: auto;
}

.generate {
    min-height: 52px;
    font-size: 1.08rem;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Anime Image Generator", css=CSS) as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown("# Anime Image Generator\nCreate anime-style artwork with remote GPU inference.")
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the image...", lines=5, max_lines=10)
            negative = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)
            with gr.Accordion("Advanced prompt settings", open=False):
                with gr.Row():
                    style = gr.Dropdown(
                        ["Anime", "Detailed anime", "Cinematic anime", "Illustration", "Manga-inspired"],
                        value="Detailed anime",
                        label="Style",
                    )
                    quality = gr.Checkbox(value=True, label="Quality enhancer")
                with gr.Row():
                    use_lora = gr.Checkbox(value=True, label="Use Myanime LoRA")
                    lora_strength = gr.Slider(0, 2, 0.8, step=0.05, label="LoRA strength")
            with gr.Row():
                aspect = gr.Radio(list(ASPECT_SIZES), value="Portrait", label="Aspect ratio")
                width = gr.Slider(
                    MIN_DIMENSION, MAX_DIMENSION, 832, step=DIMENSION_STEP, label="Width", interactive=False
                )
                height = gr.Slider(
                    MIN_DIMENSION, MAX_DIMENSION, 1216, step=DIMENSION_STEP, label="Height", interactive=False
                )
            with gr.Row():
                steps = gr.Slider(MIN_STEPS, MAX_STEPS, 28, step=1, label="Steps")
                guidance = gr.Slider(1, 15, 7, step=0.5, label="CFG / guidance")
                count = gr.Slider(MIN_IMAGES, MAX_IMAGES, 1, step=1, label="Images")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                random_seed = gr.Checkbox(value=True, label="Random seed")
            button = gr.Button("Generate", variant="primary", elem_classes="generate")
            gallery = gr.Gallery(label="Generated images", columns=2, object_fit="contain", height="auto")
            status = gr.Markdown("Ready.")

        aspect.change(set_aspect_size, aspect, [width, height], queue=False)
        button.click(
            generate_images,
            [
                prompt,
                negative,
                width,
                height,
                steps,
                guidance,
                seed,
                random_seed,
                count,
                style,
                quality,
                use_lora,
                lora_strength,
            ],
            [gallery, status],
        )
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=int(os.getenv("GRADIO_CONCURRENCY", "1")), max_size=20).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=False
    )
