from __future__ import annotations

import atexit
import logging
import os
from functools import lru_cache

import gradio as gr
from dotenv import load_dotenv

from generator import HuggingFaceImageGenerator, ProviderError
from utils import DEFAULT_NEGATIVE_PROMPT, enhance_prompt

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ASPECTS = {"Portrait": (832, 1216), "Square": (1024, 1024), "Landscape": (1216, 832), "Custom": None}
ALLOWED_DIMENSIONS = range(512, 1537, 64)


@lru_cache(maxsize=1)
def get_generator() -> HuggingFaceImageGenerator:
    return HuggingFaceImageGenerator(
        api_key=os.getenv("IMAGE_API_KEY", ""),
        model=os.getenv("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
        base_url=os.getenv("IMAGE_API_URL", "https://router.huggingface.co/hf-inference/models/{model}"),
        timeout=float(os.getenv("IMAGE_API_TIMEOUT", "120")),
    )


def close_generator() -> None:
    if get_generator.cache_info().currsize:
        get_generator().close()


atexit.register(close_generator)


def set_aspect(aspect: str) -> tuple[gr.Slider, gr.Slider]:
    size = ASPECTS.get(aspect)
    if size is None:
        return gr.Slider(interactive=True), gr.Slider(interactive=True)
    return gr.Slider(value=size[0], interactive=False), gr.Slider(value=size[1], interactive=False)


def generate_images(prompt: str, negative_prompt: str, width: int, height: int, steps: int,
                    guidance: float, seed: int, random_seed: bool, count: int, style: str,
                    quality: bool):
    ok, reason = prompt
    if not ok:
        raise gr.Error(reason)
    if int(width) not in ALLOWED_DIMENSIONS or int(height) not in ALLOWED_DIMENSIONS:
        raise gr.Error("Width and height must be 512–1536 pixels in steps of 64.")
    if not 1 <= int(count) <= 4 or not 10 <= int(steps) <= 50:
        raise gr.Error("Invalid image count or step count.")

    final_prompt = enhance_prompt(prompt, style, quality)
    images, seeds = [], []
    try:
        provider = get_generator()
        for index in range(int(count)):
            requested_seed = None if random_seed or int(seed) < 0 else int(seed) + index
            result = provider.generate(final_prompt, negative_prompt.strip(), int(width), int(height),
                                       int(steps), float(guidance), requested_seed)
            images.append((result.image, f"Seed {result.seed}"))
            seeds.append(result.seed)
    except (ProviderError, ValueError) as exc:
        logger.exception("Generation failed")
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected generation failure")
        raise gr.Error("Unexpected generation error. Please try again.") from exc

    details = (f"**Generated {len(images)} image(s)**  \n**Seeds:** {', '.join(map(str, seeds))}  \n"
               f"**Parameters:** {int(width)}×{int(height)}, {int(steps)} steps, CFG {float(guidance):g}  \n"
               f"**Final prompt:** {final_prompt}")
    return images, details


CSS = """
.container {max-width: 1120px !important; margin: auto;} .generate {min-height: 52px; font-size: 1.08rem;}
.notice {border-left: 3px solid #7c3aed; padding-left: 12px; color: var(--body-text-color-subdued);}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Anime Image Generator", css=CSS) as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown("# Anime Image Generator\nCreate anime-style artwork with remote GPU inference.")
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the image...", lines=5, max_lines=10)
            negative = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)
            with gr.Accordion("Advanced prompt settings", open=False):
                with gr.Row():
                    style = gr.Dropdown(list(("Anime", "Detailed anime", "Cinematic anime", "Illustration", "Manga-inspired")), value="Detailed anime", label="Style")
                    quality = gr.Checkbox(value=True, label="Quality enhancer")
            with gr.Row():
                aspect = gr.Radio(list(ASPECTS), value="Portrait", label="Aspect ratio")
                width = gr.Slider(512, 1536, 832, step=64, label="Width", interactive=False)
                height = gr.Slider(512, 1536, 1216, step=64, label="Height", interactive=False)
            with gr.Row():
                steps = gr.Slider(10, 50, 28, step=1, label="Steps")
                guidance = gr.Slider(1, 15, 7, step=0.5, label="CFG / guidance")
                count = gr.Slider(1, 4, 1, step=1, label="Images")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                random_seed = gr.Checkbox(value=True, label="Random seed")
            button = gr.Button("Generate", variant="primary", elem_classes="generate")
            gallery = gr.Gallery(label="Generated images", columns=2, object_fit="contain", height="auto")
            status = gr.Markdown("Ready.")

        aspect.change(set_aspect, aspect, [width, height], queue=False)
        button.click(generate_images, [prompt, negative, width, height, steps, guidance, seed,
                                      random_seed, count, style, quality], [gallery, status])
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=int(os.getenv("GRADIO_CONCURRENCY", "1")), max_size=20).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=False
    )
