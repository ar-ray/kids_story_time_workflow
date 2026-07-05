"""CLI.

  python -m kids_story_pipeline run --story examples/sample_story.txt --mock
  python -m kids_story_pipeline resume <RUN_ID> --approve
  python -m kids_story_pipeline doctor
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .config import load_profile
from .graph import GatePaused, resume_run, start_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kids_story_pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline on a story file")
    p_run.add_argument("--story", required=True, type=Path)
    p_run.add_argument("--profile", default="bedtime")
    p_run.add_argument("--mock", action="store_true",
                       help="use mock providers (no API keys, no cost)")
    p_run.add_argument("--no-gate", action="store_true",
                       help="disable the human-approval gate (fully hands-off)")

    p_res = sub.add_parser("resume", help="resume a paused/failed run")
    p_res.add_argument("run_id")
    p_res.add_argument("--profile", default="bedtime")
    p_res.add_argument("--approve", action="store_true",
                       help="approve the pending gate and continue")

    sub.add_parser("doctor", help="check environment and API keys")

    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()

    overrides = {"gate_enabled": False} if getattr(args, "no_gate", False) else None
    profile = load_profile(args.profile, overrides)

    try:
        if args.cmd == "run":
            story = args.story.read_text()
            state = start_run(story, profile, mock=args.mock)
        else:
            state = resume_run(args.run_id, profile, approve=args.approve)
    except GatePaused as gp:
        print(f"⏸  {gp}", file=sys.stderr)
        return 2

    print(f"✅ run {state.run_id} finished: {state.status}")
    print(f"   video:     {state.master_video_path}")
    print(f"   shorts:    {state.shorts_path}")
    print(f"   thumbnail: {state.thumbnail_path}")
    print(f"   metadata:  {state.metadata_path}")
    return 0


def _doctor() -> int:
    ok = True
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        print(f"{'✅' if found else '❌'} {tool}: {found or 'NOT FOUND'}")
        ok &= bool(found)
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                "ELEVENLABS_API_KEY", "FAL_KEY"):
        present = bool(os.environ.get(key))
        print(f"{'✅' if present else '⚠️ '} {key}: "
              f"{'set' if present else 'missing (mock mode still works)'}")
    print("Mock mode needs only ffmpeg/ffprobe. Real mode needs all four keys.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
