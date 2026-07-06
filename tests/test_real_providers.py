"""Offline tests for the real provider wrappers and the smoke runner.

All network calls are stubbed by monkeypatching `requests` inside
kids_story_pipeline.providers.real — nothing here talks to the internet.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from kids_story_pipeline.config import load_profile
from kids_story_pipeline.providers import real
from kids_story_pipeline import smoke


class FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200,
                 text="", url="https://api.example/x"):
        self._payload = payload
        self.content = content
        self.status_code = status_code
        self.text = text
        self.url = url

    def json(self):
        return self._payload


@pytest.fixture()
def profile(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                "ELEVENLABS_API_KEY", "FAL_KEY"):
        monkeypatch.setenv(key, f"test-{key.lower()}")
    return load_profile("bedtime")


def test_http_errors_carry_response_body():
    """Bare '403 Forbidden' hides the actionable detail (billing, quota...)
    — the raised error must include the provider's response body."""
    import requests as _requests
    resp = FakeResponse(status_code=403,
                        text='{"detail":"User is locked. Exhausted balance."}')
    with pytest.raises(_requests.HTTPError, match="Exhausted balance"):
        real._raise_for_status(resp)
    real._raise_for_status(FakeResponse(status_code=200))  # no raise


# ---- Anthropic ---------------------------------------------------------------

def test_anthropic_parses_fenced_json_and_sends_headers(profile, monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return FakeResponse(payload={"content": [
            {"type": "text", "text": '```json\n{"ok": true}\n```'}]})

    monkeypatch.setattr(real.requests, "post", fake_post)
    out = real.AnthropicLLM(profile).complete_json("sys", "prompt")
    assert out == {"ok": True}
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["headers"]["x-api-key"] == "test-anthropic_api_key"
    assert captured["body"]["model"] == "claude-sonnet-4-6"


# ---- Gemini images -----------------------------------------------------------

def test_gemini_uses_header_auth_and_new_model_id(profile, tmp_path,
                                                  monkeypatch):
    from io import BytesIO
    from PIL import Image
    captured = {}
    # Gemini returns its native resolution (1376x768 for 16:9 at 1K)
    buf = BytesIO()
    Image.new("RGB", (1376, 768), (10, 20, 40)).save(buf, "PNG")
    png = base64.b64encode(buf.getvalue()).decode()

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return FakeResponse(payload={"candidates": [{"content": {"parts": [
            {"inlineData": {"data": png}}]}}]})

    monkeypatch.setattr(real.requests, "post", fake_post)
    out = tmp_path / "img.png"
    real.GeminiImages(profile).generate("a star", out, size=(1280, 720))

    assert "gemini-3-pro-image:generateContent" in captured["url"]
    assert "key=" not in captured["url"]            # key must not leak into URL
    assert captured["headers"]["x-goog-api-key"] == "test-gemini_api_key"
    cfg = captured["body"]["generationConfig"]["imageConfig"]
    assert cfg["aspectRatio"] == "16:9"
    assert Image.open(out).size == (1280, 720)      # resized to requested size


@pytest.mark.parametrize("size,expected", [
    ((1280, 720), "16:9"), ((720, 1280), "9:16"), ((512, 512), "1:1")])
def test_gemini_aspect_ratio_from_size(size, expected):
    assert real.GeminiImages._aspect_ratio(size) == expected


# ---- Kling via fal -----------------------------------------------------------

def _kling_fakes(monkeypatch, captured):
    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(submit_url=url, headers=headers, body=json)
        return FakeResponse(payload={
            "status_url": "https://queue.fal.run/x/status",
            "response_url": "https://queue.fal.run/x/response"})

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/status"):
            return FakeResponse(payload={"status": "COMPLETED"})
        if url.endswith("/response"):
            return FakeResponse(payload={
                "video": {"url": "https://cdn.fal.example/out.mp4"}})
        return FakeResponse(content=b"fake-mp4-bytes")

    monkeypatch.setattr(real.requests, "post", fake_post)
    monkeypatch.setattr(real.requests, "get", fake_get)


@pytest.mark.parametrize("duration_s,expected", [
    (2.0, "3"), (5.0, "5"), (20.0, "15")])
def test_kling_payload_schema_and_duration_clamp(profile, tmp_path,
                                                 monkeypatch, duration_s,
                                                 expected):
    captured = {}
    _kling_fakes(monkeypatch, captured)
    img = tmp_path / "seed.png"
    img.write_bytes(b"png")
    out = tmp_path / "clip.mp4"

    real.KlingVideo(profile).animate(img, "gentle zoom", duration_s, out)

    body = captured["body"]
    assert "image_url" not in body                   # renamed in Kling v3
    assert body["start_image_url"].startswith("data:image/png;base64,")
    assert body["duration"] == expected
    assert body["generate_audio"] is False
    assert captured["headers"]["Authorization"] == "Key test-fal_key"
    assert out.read_bytes() == b"fake-mp4-bytes"


def test_kling_download_retries_flaky_connection(profile, tmp_path,
                                                 monkeypatch):
    captured = {}
    attempts = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(body=json)
        return FakeResponse(payload={
            "status_url": "https://queue.fal.run/x/status",
            "response_url": "https://queue.fal.run/x/response"})

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/status"):
            return FakeResponse(payload={"status": "COMPLETED"})
        if url.endswith("/response"):
            return FakeResponse(payload={
                "video": {"url": "https://cdn.fal.example/out.mp4"}})
        attempts["n"] += 1
        if attempts["n"] == 1:  # first download dies mid-stream
            raise real.requests.exceptions.ChunkedEncodingError("broken")
        return FakeResponse(content=b"mp4-bytes-after-retry")

    monkeypatch.setattr(real.requests, "post", fake_post)
    monkeypatch.setattr(real.requests, "get", fake_get)
    monkeypatch.setattr(real.time, "sleep", lambda s: None)
    img = tmp_path / "seed.png"
    img.write_bytes(b"png")
    out = tmp_path / "clip.mp4"
    real.KlingVideo(profile).animate(img, "zoom", 5.0, out)
    assert attempts["n"] == 2
    assert out.read_bytes() == b"mp4-bytes-after-retry"


# ---- ElevenLabs --------------------------------------------------------------

def test_elevenlabs_sfx_and_music_clamp_durations(profile, tmp_path,
                                                  monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return FakeResponse(content=b"audio-bytes")

    monkeypatch.setattr(real.requests, "post", fake_post)
    audio = real.ElevenLabsAudio(profile)

    audio.sfx("wind", 0.1, tmp_path / "a.mp3")       # below API minimum
    audio.sfx("wind", 45.0, tmp_path / "b.mp3")      # above API maximum
    audio.music("lullaby", 1.0, tmp_path / "c.mp3")  # below 3000 ms floor
    audio.music("lullaby", 700.0, tmp_path / "d.mp3")  # above 600000 ms cap

    assert calls[0][1]["duration_seconds"] == 0.5
    assert calls[1][1]["duration_seconds"] == 30.0
    assert calls[2][1]["music_length_ms"] == 3000
    assert calls[3][1]["music_length_ms"] == 600_000


def test_elevenlabs_first_premade_voice(profile, monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        assert url.endswith("/v1/voices")
        return FakeResponse(payload={"voices": [
            {"voice_id": "libX", "category": "generated"},
            {"voice_id": "preY", "category": "premade"}]})

    monkeypatch.setattr(real.requests, "get", fake_get)
    assert real.ElevenLabsAudio(profile).first_premade_voice() == "preY"


def test_elevenlabs_tts_endpoint_and_settings(profile, tmp_path, monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return FakeResponse(content=b"speech-bytes")

    monkeypatch.setattr(real.requests, "post", fake_post)
    out = tmp_path / "v.mp3"
    raw_id = "VoiceId0123456789abc"  # raw ids (16+ alnum) pass through
    real.ElevenLabsAudio(profile).tts("hi", raw_id, out)
    assert captured["url"].endswith(f"/v1/text-to-speech/{raw_id}")
    assert captured["headers"]["xi-api-key"] == "test-elevenlabs_api_key"
    assert captured["body"]["model_id"] == "eleven_multilingual_v2"
    # pacing comes from the profile (tts_speed), defaulting to 0.85
    assert captured["body"]["voice_settings"]["speed"] == 0.85
    assert out.read_bytes() == b"speech-bytes"


def test_elevenlabs_unmapped_role_reads_as_narrator(profile, tmp_path,
                                                    monkeypatch):
    """The script LLM invents roles like 'refrain' — they must fall back to
    the narrator voice, not 404 as a literal voice id."""
    urls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        urls.append(url)
        return FakeResponse(content=b"speech-bytes")

    monkeypatch.setattr(real.requests, "post", fake_post)
    audio = real.ElevenLabsAudio(profile)
    narrator_id = profile.voices["narrator"]

    audio.tts("hi", "refrain", tmp_path / "a.mp3")     # unmapped role
    audio.tts("hi", "narrator", tmp_path / "b.mp3")    # mapped role
    audio.tts("hi", "conductor", tmp_path / "c.mp3")   # mapped role
    assert urls[0].endswith(f"/v1/text-to-speech/{narrator_id}")
    assert urls[1].endswith(f"/v1/text-to-speech/{narrator_id}")
    assert urls[2].endswith(f"/v1/text-to-speech/{profile.voices['conductor']}")


# ---- smoke runner ------------------------------------------------------------

def test_resolve_smoke_voice_prefers_configured_profile_voice():
    prof = load_profile("bedtime")  # now ships real premade voice ids
    voice, fallback = smoke.resolve_smoke_voice(prof)
    assert not fallback and not voice.startswith(smoke.PLACEHOLDER_PREFIX)


def test_resolve_smoke_voice_falls_back_to_account_premade():
    prof = load_profile("bedtime",
                        overrides={"voices": {"narrator": "REPLACE_WITH_X"}})
    voice, fallback = smoke.resolve_smoke_voice(prof, _FakeAudio())
    assert fallback and voice == "premade-fallback"

    with pytest.raises(RuntimeError, match="no narrator voice"):
        smoke.resolve_smoke_voice(prof, audio=None)


class _FakeLLM:
    def complete_json(self, system, prompt):
        return {"ok": True}


class _FakeImages:
    def generate(self, prompt, out, reference=None, size=(1280, 720)):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"img")
        return out


class _FakeAudio:
    def tts(self, text, voice, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"aud")
        return out

    def first_premade_voice(self):
        return "premade-fallback"


def _video_missing_key():
    raise real.MissingKeyError("FAL_KEY is not set")


def _factories():
    return {"llm": _FakeLLM, "image": _FakeImages, "tts": _FakeAudio,
            "video": _video_missing_key}


def test_smoke_only_subset_passes_and_notes_fallback_voice(tmp_path, capsys):
    prof = load_profile("bedtime",
                        overrides={"voices": {"narrator": "REPLACE_WITH_X"}})
    ok = smoke.run_smoke(_factories(), prof, tmp_path,
                         only={"llm", "image", "tts"})
    out = capsys.readouterr().out
    assert ok
    assert "✅ llm" in out and "✅ image" in out and "✅ tts" in out
    assert "video" not in out          # deselected check never constructed
    assert "premade voice as fallback" in out


def test_smoke_one_failure_does_not_hide_others(tmp_path, capsys):
    prof = load_profile("bedtime")
    ok = smoke.run_smoke(_factories(), prof, tmp_path)
    out = capsys.readouterr().out
    assert not ok
    assert "✅ llm" in out and "✅ image" in out and "✅ tts" in out
    assert "❌ video" in out and "FAL_KEY" in out
    assert "❌ smoke failed" in out
