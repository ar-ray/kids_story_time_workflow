"""Real provider clients. Thin `requests` wrappers, keys from environment.

Endpoints/models verified against provider docs 2026-07 (Anthropic Messages,
Gemini generateContent image API, fal.ai queue + Kling v3 schema, ElevenLabs
TTS/SFX/music). Tests are offline by design, so re-verify before a paid run
if providers have shipped changes since:
  - Anthropic:   https://docs.claude.com/en/api/overview
  - Gemini img:  https://ai.google.dev/gemini-api/docs/image-generation
  - ElevenLabs:  https://elevenlabs.io/docs/api-reference
  - fal (Kling): https://fal.ai/models (kling image-to-video)
Run `python -m kids_story_pipeline doctor` to sanity-check keys.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

from . import LLMProvider, ImageProvider, VideoProvider, AudioProvider


class MissingKeyError(RuntimeError):
    pass


def _raise_for_status(resp) -> None:
    """Like requests' raise_for_status, but with the response body in the
    message — provider error bodies carry the actionable detail (quota
    exhausted, billing required, plan restrictions...)."""
    if resp.status_code >= 400:
        body = (resp.text or "").strip()[:300]
        raise requests.HTTPError(
            f"{resp.status_code} error for {resp.url}: {body}", response=resp)


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
        _raise_for_status(resp)
        text = "".join(b.get("text", "") for b in resp.json()["content"]
                       if b.get("type") == "text")
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return json.loads(text)


class GeminiImages(ImageProvider):
    """Nano Banana Pro (Gemini image model) via the Gemini API."""

    def __init__(self, profile):
        self.model = profile.image_model
        self.api_key = _key("GEMINI_API_KEY")

    @staticmethod
    def _aspect_ratio(size: tuple[int, int]) -> str:
        w, h = size
        if w == h:
            return "1:1"
        return "16:9" if w > h else "9:16"

    def generate(self, prompt: str, out: Path, reference: Path | None = None,
                 size: tuple[int, int] = (1280, 720)) -> Path:
        import base64
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent")
        parts: list[dict] = [{"text": prompt}]
        if reference is not None:
            parts.insert(0, {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(reference.read_bytes()).decode(),
                }
            })
        resp = requests.post(
            url,
            headers={"x-goog-api-key": self.api_key},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "imageConfig": {"aspectRatio": self._aspect_ratio(size)},
                },
            },
            timeout=300,
        )
        _raise_for_status(resp)
        for cand in resp.json().get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                data = part.get("inlineData") or part.get("inline_data")
                if data and data.get("data"):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(base64.b64decode(data["data"]))
                    self._fit_to_size(out, size)
                    return out
        raise RuntimeError(f"Gemini returned no image for prompt: {prompt[:80]}")

    @staticmethod
    def _fit_to_size(path: Path, size: tuple[int, int]) -> None:
        """Gemini returns its native resolution (e.g. 1376x768 for 16:9);
        honor the requested size so downstream QC/ffmpeg get exact frames."""
        from PIL import Image
        with Image.open(path) as im:
            if im.size == size:
                return
            im.convert("RGB").resize(size, Image.LANCZOS).save(path)


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
        # Kling v3 accepts duration as a string enum "3".."15" only.
        duration = str(min(15, max(3, int(round(duration_s)))))
        submit = requests.post(
            f"https://queue.fal.run/{self.model}",
            headers={"Authorization": f"Key {self.api_key}"},
            json={"start_image_url": data_uri, "prompt": motion_prompt,
                  "duration": duration,
                  # the pipeline supplies its own narration/music tracks
                  "generate_audio": False},
            timeout=60,
        )
        _raise_for_status(submit)
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
        resp = requests.get(response_url,
                            headers={"Authorization": f"Key {self.api_key}"},
                            timeout=60)
        # failed jobs also reach status COMPLETED — the error is in the
        # response body (e.g. 422 image_too_small)
        _raise_for_status(resp)
        result = resp.json()
        if not result.get("video", {}).get("url"):
            raise RuntimeError(f"Kling returned no video: {str(result)[:300]}")
        video_url = result["video"]["url"]
        out.parent.mkdir(parents=True, exist_ok=True)
        # the rendered clip is already paid for — retry a flaky download
        # (~27MB transfers can die with ChunkedEncodingError mid-stream)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                out.write_bytes(requests.get(video_url, timeout=300).content)
                return out
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                last_exc = exc
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(
            f"Kling clip download failed after 3 attempts: {last_exc}")


class ElevenLabsAudio(AudioProvider):
    BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, profile):
        self.api_key = _key("ELEVENLABS_API_KEY")
        self.voices = profile.voices or {}
        self.speed = float((profile.raw or {}).get("tts_speed", 0.85))

    def _headers(self) -> dict:
        return {"xi-api-key": self.api_key}

    def _resolve_voice(self, voice: str) -> str:
        """Map a script role to a voice id. Unmapped roles (the LLM invents
        ones like 'refrain' or character names) read in the narrator's voice
        instead of 404ing as a literal voice id."""
        mapped = self.voices.get(voice)
        if mapped:
            return mapped
        if re.fullmatch(r"[A-Za-z0-9]{16,}", voice):
            return voice  # already a raw ElevenLabs voice id
        return self.voices.get("narrator", voice)

    def first_premade_voice(self) -> str | None:
        """First premade voice on the account. Free plans can use premade
        voices via the API but NOT library voices (402 paid_plan_required)."""
        resp = requests.get(f"{self.BASE}/voices", headers=self._headers(),
                            timeout=60)
        _raise_for_status(resp)
        for v in resp.json().get("voices", []):
            if v.get("category") == "premade":
                return v["voice_id"]
        return None

    def tts(self, text: str, voice: str, out: Path) -> Path:
        voice_id = self._resolve_voice(voice)
        resp = requests.post(
            f"{self.BASE}/text-to-speech/{voice_id}",
            headers=self._headers(),
            json={"text": text, "model_id": "eleven_multilingual_v2",
                  "voice_settings": {"stability": 0.65, "similarity_boost": 0.8,
                                     "style": 0.15, "speed": self.speed}},
            timeout=300,
        )
        _raise_for_status(resp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return out

    def sfx(self, prompt: str, duration_s: float, out: Path) -> Path:
        resp = requests.post(
            f"{self.BASE}/sound-generation", headers=self._headers(),
            # API allows 0.5-30 s
            json={"text": prompt,
                  "duration_seconds": max(0.5, min(30.0, duration_s))},
            timeout=300,
        )
        _raise_for_status(resp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return out

    def music(self, prompt: str, duration_s: float, out: Path) -> Path:
        resp = requests.post(
            f"{self.BASE}/music", headers=self._headers(),
            # API allows 3000-600000 ms
            json={"prompt": prompt,
                  "music_length_ms": max(3000, min(600_000,
                                                   int(duration_s * 1000)))},
            timeout=600,
        )
        _raise_for_status(resp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return out
