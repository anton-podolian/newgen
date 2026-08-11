from __future__ import annotations

import io
import logging
import random
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedImage:
    image: Image.Image
    seed: int


class ProviderError(RuntimeError):
    """An error that is safe to show to an app user."""


class HuggingFaceImageGenerator:
    """Text-to-image through Hugging Face's routed hf-inference HTTP API."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 120.0) -> None:
        self.model = model.strip()
        if not api_key.strip():
            raise ValueError("IMAGE_API_KEY is not configured.")
        if not self.model:
            raise ValueError("IMAGE_MODEL is not configured.")
        endpoint = base_url.rstrip("/")
        if "{model}" in endpoint:
            endpoint = endpoint.format(model=quote(self.model, safe="/"))
        self.endpoint = endpoint
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key.strip()}", "Accept": "image/png"},
            timeout=httpx.Timeout(timeout, connect=15.0),
            follow_redirects=True,
        )

    def generate(self, prompt: str, negative_prompt: str, width: int, height: int,
                 steps: int, guidance_scale: float, seed: int | None) -> GeneratedImage:
        used_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        parameters: dict[str, Any] = {
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": used_seed,
        }
        try:
            response = self.client.post(self.endpoint, json={"inputs": prompt, "parameters": parameters})
        except httpx.TimeoutException as exc:
            raise ProviderError("The image service timed out. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("The image service is currently unavailable.") from exc

        if response.status_code in (401, 403):
            raise ProviderError("The image service rejected its API credentials.")
        if response.status_code == 429:
            raise ProviderError("The image service rate limit was reached. Please retry later.")
        if response.status_code >= 400:
            detail = self._safe_error(response)
            logger.error("Inference API error %s: %s", response.status_code, detail)
            raise ProviderError(f"Image generation failed (provider status {response.status_code}).")
        if not response.content:
            raise ProviderError("The image service returned no image.")
        try:
            image = Image.open(io.BytesIO(response.content))
            image.load()
            return GeneratedImage(image=image.convert("RGB"), seed=used_seed)
        except (UnidentifiedImageError, OSError) as exc:
            logger.error("Unexpected inference response content-type=%s", response.headers.get("content-type"))
            raise ProviderError("The image service returned an invalid image response.") from exc

    @staticmethod
    def _safe_error(response: httpx.Response) -> str:
        try:
            data = response.json()
            return str(data.get("error", data))[:500] if isinstance(data, dict) else str(data)[:500]
        except ValueError:
            return response.text[:500]

    def close(self) -> None:
        self.client.close()
