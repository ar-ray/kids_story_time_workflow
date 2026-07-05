"""Smoke checks: one tiny real API call per provider to confirm keys work.

Costs real money (cents for llm/tts, ~$0.15 for the image, tens of cents for
the 3 s Kling clip). Run before the first full paid pipeline run:

  python -m kids_story_pipeline smoke
  python -m kids_story_pipeline smoke --only llm,tts
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable

from .providers.real import MissingKeyError

CHECKS = ("llm", "image", "tts", "video")

# ElevenLabs premade voice ("Rachel") so tts can be smoked before the
# channel's own voice ids are configured in the profile.
FALLBACK_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
PLACEHOLDER_PREFIX = "REPLACE_WITH"


def resolve_smoke_voice(profile) -> tuple[str, bool]:
    """Return (voice_id, used_fallback) for the narrator voice."""
    configured = (profile.voices or {}).get("narrator", "")
    if not configured or configured.startswith(PLACEHOLDER_PREFIX):
        return FALLBACK_VOICE_ID, True
    return configured, False


def real_factories(profile) -> dict[str, Callable[[], object]]:
    """Lazy constructors so `--only llm` needs only the Anthropic key."""
    from .providers.real import (AnthropicLLM, ElevenLabsAudio, GeminiImages,
                                 KlingVideo)
    return {
        "llm": lambda: AnthropicLLM(profile),
        "image": lambda: GeminiImages(profile),
        "tts": lambda: ElevenLabsAudio(profile),
        "video": lambda: KlingVideo(profile),
    }


def _tiny_png(out: Path) -> Path:
    from PIL import Image
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), (20, 30, 70)).save(out)
    return out


def run_smoke(factories: dict[str, Callable[[], object]], profile,
              out_dir: Path, only: set[str] | None = None) -> bool:
    """Run one minimal call per selected provider. Returns True if all passed.

    `factories` maps check name -> zero-arg constructor for that provider
    (see `real_factories`); construction happens inside each check so a
    missing key fails only its own check.
    """
    selected = [c for c in CHECKS if only is None or c in only]
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[bool, str]] = {}

    def attempt(name: str, fn):
        t0 = time.time()
        try:
            detail = fn(factories[name]())
            results[name] = (True, f"{detail} ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - report every failure kind
            results[name] = (False, f"{type(exc).__name__}: {exc}")
            if not isinstance(exc, MissingKeyError):
                traceback.print_exc()

    def check_llm(llm):
        data = llm.complete_json(
            "You are a health check. Respond with ONLY a JSON object.",
            'Return exactly {"ok": true}')
        if data.get("ok") is not True:
            raise RuntimeError(f"unexpected JSON: {data}")
        return f"model {profile.llm_model} returned valid JSON"

    def check_image(images):
        out = out_dir / "smoke_image.png"
        images.generate(
            "a single small gold star in a dark blue night sky, "
            + profile.raw.get("style_anchor", ""),
            out, size=profile.size)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError("no image bytes written")
        return f"{profile.image_model} -> {out}"

    def check_tts(audio):
        voice, used_fallback = resolve_smoke_voice(profile)
        out = out_dir / "smoke_tts.mp3"
        audio.tts("Hello from the bedtime pipeline.", voice, out)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError("no audio bytes written")
        note = (" [using premade fallback voice — set narrator/conductor "
                "ids in the profile]" if used_fallback else "")
        return f"voice {voice} -> {out}{note}"

    def check_video(video):
        img = _tiny_png(out_dir / "smoke_video_seed.png")
        out = out_dir / "smoke_video.mp4"
        video.animate(img, "slow gentle zoom, stars twinkling", 3.0, out)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError("no video bytes written")
        return f"{profile.video_model} -> {out}"

    checks = {"llm": check_llm, "image": check_image,
              "tts": check_tts, "video": check_video}
    for name in selected:
        attempt(name, checks[name])

    all_ok = True
    for name in selected:
        ok, detail = results[name]
        all_ok &= ok
        print(f"{'✅' if ok else '❌'} {name}: {detail}")
    print(f"{'✅ all smoke checks passed' if all_ok else '❌ smoke failed'} "
          f"— artifacts in {out_dir}")
    return all_ok
