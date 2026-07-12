"""Mock providers: deterministic fixtures, zero network, zero cost.

The point of mock mode is that the ENTIRE pipeline (including real ffmpeg
assembly) runs end-to-end so orchestration, timing math, and rendering are
exercised for real; only paid external APIs are substituted.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image, ImageDraw

from . import LLMProvider, ImageProvider, VideoProvider, AudioProvider
from .. import ffmpeg_utils as ff

_WORD = re.compile(r"[A-Za-z']+")

# words-per-second used by mock TTS; mirrors bedtime pace (~105 wpm)
_MOCK_WPS = 105 / 60.0


class MockLLM(LLMProvider):
    """Derives structured output from the input deterministically."""

    def __init__(self, profile):
        self.profile = profile

    def complete_json(self, system: str, prompt: str, images=None) -> dict:
        if "SCRIPT_TASK" in system:
            return self._script(prompt)
        if "SCENE_TASK" in system:
            return self._scenes(prompt)
        if "VISION_QC_TASK" in system:
            return {"matches": True, "issues": [],
                    "corrected_prompt": ""}
        if "PERSONA_TASK" in system:
            return {"kid_score": 0.92, "parent_score": 0.95,
                    "comments": ["mock persona review: cozy and clear"]}
        if "PACKAGE_TASK" in system:
            return {"description": "A gentle bedtime story to help little ones drift off.",
                    "tags": ["bedtime story", "sleep story for kids", "calming story"]}
        return {}

    def _script(self, prompt: str) -> dict:
        story = prompt.split("STORY:", 1)[-1].strip()
        paras = [p.strip() for p in story.split("\n\n") if p.strip()]
        title = paras[0].splitlines()[0][:60] if paras else "A Sleepy Story"
        lines = []
        for p in paras:
            for sent in re.split(r"(?<=[.!?])\s+", p.replace("\n", " ")):
                sent = sent.strip()
                if not sent:
                    continue
                role = "conductor" if sent.startswith('"') else "narrator"
                lines.append({"text": sent, "role": role})
        return {"title": title, "refrain": "Around and around went the night train.",
                "lines": lines}

    def _scenes(self, prompt: str) -> dict:
        n_lines = prompt.count("\n") + 1
        per_scene = 4
        n_scenes = max(3, min(16, (n_lines + per_scene - 1) // per_scene))
        scenes = []
        for i in range(n_scenes):
            scenes.append({
                "title": f"Scene {i + 1}",
                "line_start": i * per_scene,
                "line_end": min(n_lines, (i + 1) * per_scene),
                "image_prompt": f"storybook scene {i + 1}, cozy night palette",
                "sfx_prompt": "soft night ambience",
            })
        return {"scenes": scenes}


_PALETTE = [(41, 50, 92), (52, 73, 102), (70, 60, 100), (36, 66, 84),
            (60, 48, 88), (44, 78, 96), (80, 64, 72), (34, 56, 78)]


class MockImages(ImageProvider):
    def generate(self, prompt: str, out: Path, reference: Path | None = None,
                 size: tuple[int, int] = (1280, 720)) -> Path:
        idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(_PALETTE)
        img = Image.new("RGB", size, _PALETTE[idx])
        d = ImageDraw.Draw(img)
        # soft "moon" so Ken Burns motion is visible in mock renders
        w, h = size
        d.ellipse([w * 0.72, h * 0.12, w * 0.86, h * 0.36], fill=(236, 229, 200))
        d.text((40, h - 60), prompt[:70], fill=(235, 235, 235))
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        return out


class MockVideo(VideoProvider):
    """Stands in for Kling: produces a real (slightly stronger-zoom) clip."""

    def animate(self, image: Path, motion_prompt: str, duration_s: float,
                out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        return ff.make_kenburns_clip(image, duration_s, out,
                                     zoom_rate=0.0012, max_zoom=1.15)


class MockAudio(AudioProvider):
    def __init__(self, profile):
        self.profile = profile

    def tts(self, text: str, voice: str, out: Path) -> Path:
        words = len(_WORD.findall(text)) or 1
        dur = words / _MOCK_WPS
        freq = 200 if voice == "narrator" else 320
        out.parent.mkdir(parents=True, exist_ok=True)
        return ff.synth_tone(out, dur, freq=freq)

    def sfx(self, prompt: str, duration_s: float, out: Path) -> Path:
        return ff.synth_tone(out, duration_s, freq=90)

    def music(self, prompt: str, duration_s: float, out: Path) -> Path:
        return ff.synth_tone(out, duration_s, freq=110)
