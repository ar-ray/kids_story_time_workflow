"""Provider interfaces. Mock and real implementations share these contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class LLMProvider(ABC):
    @abstractmethod
    def complete_json(self, system: str, prompt: str) -> dict:
        """Return a JSON object from the model."""


class ImageProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, out: Path, reference: Path | None = None,
                 size: tuple[int, int] = (1280, 720)) -> Path: ...


class VideoProvider(ABC):
    @abstractmethod
    def animate(self, image: Path, motion_prompt: str, duration_s: float,
                out: Path) -> Path:
        """Image-to-video for hero scenes."""


class AudioProvider(ABC):
    @abstractmethod
    def tts(self, text: str, voice: str, out: Path) -> Path: ...

    @abstractmethod
    def sfx(self, prompt: str, duration_s: float, out: Path) -> Path: ...

    @abstractmethod
    def music(self, prompt: str, duration_s: float, out: Path) -> Path: ...


class Providers:
    def __init__(self, llm: LLMProvider, images: ImageProvider,
                 video: VideoProvider, audio: AudioProvider):
        self.llm = llm
        self.images = images
        self.video = video
        self.audio = audio


def build_providers(mock: bool, profile) -> Providers:
    if mock:
        from .mock import MockLLM, MockImages, MockVideo, MockAudio
        return Providers(MockLLM(profile), MockImages(), MockVideo(), MockAudio(profile))
    from .real import AnthropicLLM, GeminiImages, KlingVideo, ElevenLabsAudio
    return Providers(AnthropicLLM(profile), GeminiImages(profile),
                     KlingVideo(profile), ElevenLabsAudio(profile))
