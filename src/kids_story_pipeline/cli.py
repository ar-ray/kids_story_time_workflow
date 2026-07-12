"""CLI.

  python -m kids_story_pipeline run --story examples/sample_story.txt --mock
  python -m kids_story_pipeline resume <RUN_ID> --approve
  python -m kids_story_pipeline doctor
  python -m kids_story_pipeline smoke [--only llm,image,tts,video]
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
    p_run.add_argument("--run-id", default=None,
                       help="name the run folder (default: timestamp id); "
                            "assets land in runs/<run-id>/")
    p_run.add_argument("--out", type=Path, default=None,
                       help="also copy the final video here (directory -> "
                            "<story-stem>.mp4 inside it)")
    p_run.add_argument("--mock", action="store_true",
                       help="use mock providers (no API keys, no cost)")
    p_run.add_argument("--no-gate", action="store_true",
                       help="disable the human-approval gate (fully hands-off)")

    p_res = sub.add_parser("resume", help="resume a paused/failed run")
    p_res.add_argument("run_id")
    p_res.add_argument("--profile", default="bedtime")
    p_res.add_argument("--out", type=Path, default=None,
                       help="also copy the final video here (directory -> "
                            "<run-id>.mp4 inside it)")
    p_res.add_argument("--approve", action="store_true",
                       help="approve the pending gate and continue")

    sub.add_parser("doctor", help="check environment and API keys")

    p_smoke = sub.add_parser(
        "smoke", help="one tiny REAL (paid) API call per provider to "
                      "confirm keys work before a full run")
    p_smoke.add_argument("--profile", default="bedtime")
    p_smoke.add_argument("--only", default=None,
                         help="comma-separated subset of: llm,image,tts,video")

    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()

    if args.cmd == "smoke":
        return _smoke(args)

    overrides = {"gate_enabled": False} if getattr(args, "no_gate", False) else None
    profile = load_profile(args.profile, overrides)

    try:
        if args.cmd == "run":
            story = args.story.read_text()
            state = start_run(story, profile, mock=args.mock,
                              run_id=args.run_id)
            stem = args.story.stem
        else:
            state = resume_run(args.run_id, profile, approve=args.approve)
            stem = args.run_id
    except GatePaused as gp:
        print(f"⏸  {gp}", file=sys.stderr)
        return 2

    print(f"✅ run {state.run_id} finished: {state.status}")
    print(f"   video:     {state.master_video_path}")
    print(f"   shorts:    {state.shorts_path}")
    print(f"   thumbnail: {state.thumbnail_path}")
    print(f"   metadata:  {state.metadata_path}")
    if args.out and state.master_video_path:
        dest = _deliver(Path(state.master_video_path), args.out, stem)
        print(f"   delivered: {dest}")
    return 0


def _deliver(master: Path, out: Path, stem: str) -> Path:
    """Copy the final video to `out` (a directory gets <stem>.mp4 inside)."""
    dest = out / f"{stem}.mp4" if (out.is_dir() or not out.suffix) else out
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(master, dest)
    return dest


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
    print("Keys set? Run `python -m kids_story_pipeline smoke` to make one "
          "tiny paid call per provider before a full run.")
    return 0 if ok else 1


def _smoke(args) -> int:
    from datetime import datetime

    from .smoke import CHECKS, real_factories, run_smoke

    only = None
    if args.only:
        only = {c.strip() for c in args.only.split(",") if c.strip()}
        unknown = only - set(CHECKS)
        if unknown:
            print(f"unknown --only checks: {', '.join(sorted(unknown))} "
                  f"(valid: {','.join(CHECKS)})", file=sys.stderr)
            return 2

    profile = load_profile(args.profile)
    out_dir = (Path("runs")
               / f"smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    return 0 if run_smoke(real_factories(profile), profile, out_dir,
                          only=only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
