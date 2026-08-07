from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class GeneratedImage:
    image: Image.Image
    seed: int


class ProviderError(RuntimeError):
    """A safe, user-facing inference provider error."""


class ImageGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> GeneratedImage:
        raise NotImplementedError

    def close(self) -> None:
        return None
