"""Profile + environment configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class Profile:
    name: str
    target_age: str = "4-8"
    wpm: int = 105                      # bedtime narration pace
    line_pause_s: float = 0.35
    scene_pause_s: float = 0.6
    crossfade_s: float = 1.2
    outro_s: float = 20.0
    fps: int = 25
    width: int = 1280
    height: int = 720
    hero_scene_count: int = 4
    max_reading_grade: float = 3.2      # Flesch-Kincaid ceiling for ages 4-8
    scary_words: list[str] = field(default_factory=list)
    gate_threshold: float = 0.75
    gate_enabled: bool = True
    music_gain: float = 0.10
    shorts_max_s: float = 60.0
    voices: dict[str, str] = field(default_factory=dict)   # role -> voice id
    llm_model: str = "claude-sonnet-4-6"
    image_model: str = "gemini-3-pro-image"                # "Nano Banana Pro"; id verified 2026-07
    video_model: str = "fal-ai/kling-video/v3/standard/image-to-video"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


def load_profile(name: str, overrides: dict[str, Any] | None = None) -> Profile:
    path = CONFIG_DIR / "profiles" / f"{name}.yaml"
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    if overrides:
        data.update(overrides)
    known = {k: v for k, v in data.items() if k in Profile.__dataclass_fields__}
    prof = Profile(name=name, **known)
    prof.raw = data
    return prof


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)
