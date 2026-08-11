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
    """SDXL client routed to fal's serverless API through Hugging Face."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 120.0) -> None:
        self.model = model.strip()
        if not api_key.strip():
            raise ValueError("HF_TOKEN is not configured.")
        if not self.model:
            raise ValueError("IMAGE_MODEL is not configured.")
        endpoint = base_url.rstrip("/")
        if "{model}" in endpoint:
            endpoint = endpoint.format(model=quote(self.model, safe="/"))
        self.endpoint = endpoint
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key.strip()}", "Accept": "application/json"},
            timeout=httpx.Timeout(timeout, connect=15.0),
            follow_redirects=True,
        )
        # Generated-image URLs are hosted by fal's CDN. Keep the HF credential
        # on Router/Hub requests only; never forward it to the CDN.
        self.download_client = httpx.Client(
            headers={"Accept": "image/png"},
            timeout=httpx.Timeout(timeout, connect=15.0),
            follow_redirects=True,
        )

    def generate(self, prompt: str, negative_prompt: str, width: int, height: int,
                 steps: int, guidance_scale: float, seed: int | None,
                 lora_repository: str | None = None, lora_weight_name: str | None = None,
                 lora_scale: float = 1.0) -> GeneratedImage:
        used_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        # The Hugging Face fal route accepts fal's native SDXL payload.  Do not
        # wrap these fields in the generic Inference API ``parameters`` object:
        # fal expects ``prompt``, ``image_size`` and ``loras`` at the top level.
        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_size": {"width": width, "height": height},
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": used_seed,
            "enable_safety_checker": False,
        }
        if lora_repository:
            payload["loras"] = [{
                "path": self._public_lora_url(lora_repository, lora_weight_name),
                "scale": lora_scale,
            }]
        try:
            response = self.client.post(self.endpoint, json=payload)
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
        try:
            result = response.json()
            image_url = result["images"][0]["url"]
            if not isinstance(image_url, str) or not image_url:
                raise ValueError("missing image URL")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error("Unexpected fal response: %s", response.text[:500])
            raise ProviderError("The image service returned an invalid generation response.") from exc

        try:
            image_response = self.download_client.get(image_url)
            image_response.raise_for_status()
            image = Image.open(io.BytesIO(image_response.content))
            image.load()
            return GeneratedImage(image=image.convert("RGB"), seed=used_seed)
        except (httpx.HTTPError, UnidentifiedImageError, OSError) as exc:
            logger.error("Could not download generated image from fal")
            raise ProviderError("The image service returned an invalid image response.") from exc

    @staticmethod
    def _safe_error(response: httpx.Response) -> str:
        try:
            data = response.json()
            return str(data.get("error", data))[:500] if isinstance(data, dict) else str(data)[:500]
        except ValueError:
            return response.text[:500]

    def _public_lora_url(self, repository: str, weight_name: str | None) -> str:
        """Resolve the public Hub LoRA file without downloading its weights."""
        files_url = f"https://huggingface.co/api/models/{quote(repository, safe='/')}/tree/main?recursive=true"
        try:
            response = self.client.get(files_url)
            response.raise_for_status()
            files = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Could not read the public LoRA repository.") from exc

        if not isinstance(files, list):
            raise ProviderError("The LoRA repository returned an invalid file list.")
        checkpoints = [file["path"] for file in files if file.get("path", "").endswith(".safetensors")]
        if weight_name:
            if weight_name not in checkpoints:
                raise ProviderError("LORA_WEIGHT_NAME was not found in the LoRA repository.")
        elif len(checkpoints) == 1:
            weight_name = checkpoints[0]
        elif not checkpoints:
            raise ProviderError("No .safetensors file was found in the LoRA repository.")
        else:
            raise ProviderError("The LoRA repository has multiple .safetensors files. Set LORA_WEIGHT_NAME.")

        # The public adapter's live Hub mapping tells fal to use this exact
        # weight; neither Railway nor the application downloads it.
        return (
            f"https://huggingface.co/{quote(repository, safe='/')}/resolve/main/"
            f"{quote(weight_name, safe='/')}"
        )

    def close(self) -> None:
        self.client.close()
        self.download_client.close()
