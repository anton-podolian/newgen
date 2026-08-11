import io

import httpx
from PIL import Image

from generator.provider import HuggingFaceImageGenerator


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(output, format="PNG")
    return output.getvalue()


def test_fal_router_payload_contains_signed_lora_and_scale(monkeypatch) -> None:
    generator = HuggingFaceImageGenerator(
        api_key="hf_test",
        model="stabilityai/stable-diffusion-xl-base-1.0",
        base_url="https://router.huggingface.co/fal-ai/fal-ai/fast-sdxl",
    )
    monkeypatch.setattr(generator, "_public_lora_url", lambda *_: "https://huggingface.co/example/lora.safetensors")
    seen_payload = {}

    def router_handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"images": [{"url": "https://cdn.example/generated.png"}]})

    generator.client = httpx.Client(transport=httpx.MockTransport(router_handler))
    generator.download_client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=_png_bytes()))
    )

    result = generator.generate(
        prompt="anime city",
        negative_prompt="low quality",
        width=832,
        height=1216,
        steps=28,
        guidance_scale=7,
        seed=123,
        lora_repository="serenitymea/Myanime_model",
        lora_weight_name="myanime.safetensors",
        lora_scale=0.8,
    )

    assert result.seed == 123
    assert result.image.size == (1, 1)
    assert seen_payload["image_size"] == {"width": 832, "height": 1216}
    assert seen_payload["enable_safety_checker"] is False
    assert seen_payload["loras"] == [{"path": "https://huggingface.co/example/lora.safetensors", "scale": 0.8}]
