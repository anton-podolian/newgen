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
    """An error that can be shown to the person using the app."""


class HuggingFaceImageGenerator:
    """Generate SDXL images through the Hugging Face Router and fal."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 120.0) -> None:
        api_key = api_key.strip()
        model = model.strip()
        if not api_key:
            raise ValueError("HF_TOKEN is not configured.")
        if not model:
            raise ValueError("IMAGE_MODEL is not configured.")

        endpoint = base_url.rstrip("/").format(model=quote(model, safe="/"))
        request_timeout = httpx.Timeout(timeout, connect=15.0)

        self.endpoint = endpoint
        self.router = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=request_timeout,
            follow_redirects=True,
        )
        # fal stores generated images on a CDN. Do not send the Hugging Face
        # token when fetching that public image URL.
        self.images = httpx.Client(
            headers={"Accept": "image/png"},
            timeout=request_timeout,
            follow_redirects=True,
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        lora_repository: str | None = None,
        lora_weight_name: str | None = None,
        lora_scale: float = 1.0,
    ) -> GeneratedImage:
        seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        payload = self._build_payload(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            lora_repository=lora_repository,
            lora_weight_name=lora_weight_name,
            lora_scale=lora_scale,
        )

        try:
            response = self.router.post(self.endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("Image generation timed out. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("The image service is currently unavailable.") from exc

        self._raise_for_provider_error(response)
        image_url = self._image_url(response)
        return GeneratedImage(image=self._download_image(image_url), seed=seed)

    def _build_payload(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        lora_repository: str | None,
        lora_weight_name: str | None,
        lora_scale: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_size": {"width": width, "height": height},
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "enable_safety_checker": False,
        }
        if lora_repository:
            payload["loras"] = [{
                "path": self._lora_url(lora_repository, lora_weight_name),
                "scale": lora_scale,
            }]
        return payload

    def _raise_for_provider_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code in (401, 403):
            raise ProviderError("Hugging Face rejected HF_TOKEN.")
        if response.status_code == 402:
            raise ProviderError("Hugging Face included credits are exhausted. Add credits or wait for renewal.")
        if response.status_code == 429:
            raise ProviderError("The image service rate limit was reached. Please retry later.")

        detail = self._error_detail(response)
        logger.error("Image provider returned %s: %s", response.status_code, detail)
        raise ProviderError(f"Image generation failed (provider status {response.status_code}).")

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text[:500]
        return str(data.get("error", data) if isinstance(data, dict) else data)[:500]

    @staticmethod
    def _image_url(response: httpx.Response) -> str:
        try:
            image_url = response.json()["images"][0]["url"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error("Unexpected image-provider response: %s", response.text[:500])
            raise ProviderError("The image service returned an invalid generation response.") from exc
        if not isinstance(image_url, str) or not image_url:
            raise ProviderError("The image service returned an invalid image URL.")
        return image_url

    def _download_image(self, image_url: str) -> Image.Image:
        try:
            response = self.images.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image.load()
            return image.convert("RGB")
        except (httpx.HTTPError, UnidentifiedImageError, OSError) as exc:
            logger.error("Could not download the generated image.")
            raise ProviderError("The image service returned an invalid image response.") from exc

    def _lora_url(self, repository: str, weight_name: str | None) -> str:
        """Return the public safetensors URL without downloading the LoRA."""
        repository = repository.strip()
        weight_name = weight_name or self._find_single_lora_weight(repository)
        return f"https://huggingface.co/{quote(repository, safe='/')}/resolve/main/{quote(weight_name, safe='/')}"

    def _find_single_lora_weight(self, repository: str) -> str:
        files_url = f"https://huggingface.co/api/models/{quote(repository, safe='/')}/tree/main?recursive=true"
        try:
            response = self.router.get(files_url)
            response.raise_for_status()
            files = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Could not read the LoRA repository.") from exc

        weights = [entry["path"] for entry in files if entry.get("path", "").endswith(".safetensors")]
        if len(weights) == 1:
            return weights[0]
        if not weights:
            raise ProviderError("No .safetensors file was found in the LoRA repository.")
        raise ProviderError("The LoRA repository has multiple .safetensors files. Set LORA_WEIGHT_NAME.")

    def close(self) -> None:
        self.router.close()
        self.images.close()
