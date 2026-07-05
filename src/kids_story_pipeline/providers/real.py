"""Real provider clients. Thin `requests` wrappers, keys from environment.

IMPORTANT: these are written against provider docs as of mid-2026 but have NOT
been exercised against live APIs from CI. Before your first paid run, verify
endpoints/models against:
  - Anthropic:   https://docs.claude.com/en/api/overview
  - Gemini img:  https://ai.google.dev/gemini-api/docs/image-generation
  - ElevenLabs:  https://elevenlabs.io/docs/api-reference
  - fal (Kling): https://fal.ai/models (kling image-to-video)
Run `python -m kids_story_pipeline doctor` to sanity-check keys.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from . import LLMProvider, ImageProvider, VideoProvider, AudioProvider


class MissingKeyError(RuntimeError):
    pass


def _key(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise MissingKeyError(
            f"{name} is not set. Copy .env.example to .env and fill it in "
            f"(then `export $(grep -v '^#' .env | xargs)` or use direnv).")
    return val


class AnthropicLLM(LLMProvider):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, profile):
        self.model = profile.llm_model
        self.api_key = _key("ANTHROPIC_API_KEY")

    def complete_json(self, system: str, prompt: str) -> dict:
        resp = requests.post(
            self.URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4000,
                "system": system + "\nRespond with ONLY a valid JSON object.",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = "".join(b.get("text", "") for b in resp.json()["content"]
                       if b.get("type") == "text")
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return json.loads(text)


class GeminiImages(ImageProvider):
    """Nano Banana Pro (Gemini image model) via the Gemini API."""

    def __init__(self, profile):
        self.model = profile.image_model
        self.api_key = _key("GEMINI_API_KEY")

    def generate(self, prompt: str, out: Path, reference: Path | None = None,
                 size: tuple[int, int] = (1280, 720)) -> Path:
        import base64
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        parts: list[dict] = [{"text": prompt}]
        if reference is not None:
            parts.insert(0, {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(reference.read_bytes()).decode(),
                }
            })
        resp = requests.post(url, json={"contents": [{"parts": parts}]}, timeout=300)
        resp.raise_for_status()
        for cand in resp.json().get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                data = part.get("inlineData") or part.get("inline_data")
                if data and data.get("data"):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(base64.b64decode(data["data"]))
                    return out
        raise RuntimeError(f"Gemini returned no image for prompt: {prompt[:80]}")


class KlingVideo(VideoProvider):
    """Kling image-to-video via fal.ai queue API."""

    def __init__(self, profile):
        self.model = profile.video_model
        self.api_key = _key("FAL_KEY")

    def animate(self, image: Path, motion_prompt: str, duration_s: float,
                out: Path) -> Path:
        import base64
        data_uri = ("data:image/png;base64,"
                    + base64.b64encode(image.read_bytes()).decode())
        submit = requests.post(
            f"https://queue.fal.run/{self.model}",
            headers={"Authorization": f"Key {self.api_key}"},
            json={"image_url": data_uri, "prompt": motion_prompt,
                  "duration": str(int(round(duration_s)))},
            timeout=60,
        )
        submit.raise_for_status()
        status_url = submit.json()["status_url"]
        response_url = submit.json()["response_url"]
        deadline = time.time() + 900
        while time.time() < deadline:
            st = requests.get(status_url,
                              headers={"Authorization": f"Key {self.api_key}"},
                              timeout=30).json()
            if st.get("status") == "COMPLETED":
                break
            if st.get("status") in ("FAILED", "ERROR"):
                raise RuntimeError(f"Kling job failed: {st}")
            time.sleep(5)
        result = requests.get(response_url,
                              headers={"Authorization": f"Key {self.api_key}"},
                              timeout=60).json()
        video_url = result["video"]["url"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(requests.get(video_url, timeout=300).content)
        return out


class ElevenLabsAudio(AudioProvider):
    BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, profile):
        self.api_key = _key("ELEVENLABS_API_KEY")
        self.voices = profile.voices or {}

    def _headers(self) -> dict:
        return {"xi-api-key": self.api_key}

    def tts(self, text: str, voice: str, out: Path) -> Path:
        voice_id = self.voices.get(voice, voice)
        resp = requests.post(
            f"{self.BASE}/text-to-speech/{voice_id}",
            headers=self._headers(),
            json={"text": text, "model_id": "eleven_multilingual_v2",
                  "voice_settings": {"stability": 0.65, "similarity_boost": 0.8,
                                     "style": 0.15, "speed": 0.85}},
            timeout=300,
        )
        resp.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return out

    def sfx(self, prompt: str, duration_s: float, out: Path) -> Path:
        resp = requests.post(
            f"{self.BASE}/sound-generation", headers=self._headers(),
            json={"text": prompt, "duration_seconds": min(22.0, duration_s)},
            timeout=300,
        )
        resp.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return out

    def music(self, prompt: str, duration_s: float, out: Path) -> Path:
        resp = requests.post(
            f"{self.BASE}/music", headers=self._headers(),
            json={"prompt": prompt,
                  "music_length_ms": int(duration_s * 1000)},
            timeout=600,
        )
        resp.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return out
