"""Fidelity audit: re-review an existing run's imagery against the story.

REPORT-ONLY — this never generates or regenerates anything, so the only
spend is one vision review per scene (~cents) plus one per hero clip with
--clips. Use it after a run (or after editing a story) to see exactly which
scenes drifted from the story's statements, then repair ONLY what a human
confirms is wrong (delete that scene's png/clips and resume the run).

  python -m kids_story_pipeline audit <RUN_ID> [--profile ...] [--clips]
"""
from __future__ import annotations

from pathlib import Path

from .nodes import _review_clip, _review_one, _verdict_ok
from .state import PipelineState

IMAGE_COST_USD = 0.15   # rough Nano Banana Pro per image
CLIP_COST_USD = 0.35    # rough Kling 3-10s clip


class _LLMOnly:
    def __init__(self, llm):
        self.llm = llm


def audit_run(state: PipelineState, llm, run_dir: Path,
              include_clips: bool = False, printer=print) -> bool:
    """Review every scene image (and optionally hero clips). Returns True
    when everything matches the story. Never regenerates anything."""
    p = _LLMOnly(llm)
    reviews = bad_images = bad_clips = 0
    all_ok = True
    for sc in state.scenes:
        if not sc.image_path or not Path(sc.image_path).exists():
            printer(f"❌ scene {sc.id} ({sc.title}): image missing")
            all_ok = False
            continue
        v = _review_one(p, sc, state)
        reviews += 1
        ok = _verdict_ok(v)
        if ok:
            printer(f"✅ scene {sc.id} ({sc.title}): image matches the story")
        else:
            all_ok = False
            bad_images += 1
            printer(f"❌ scene {sc.id} ({sc.title}): IMAGE mismatch\n"
                    f"     observed: {v.get('observed_action', '?')[:160]}\n"
                    f"     expected: {v.get('expected_action', '?')[:160]}\n"
                    f"     issues:   {'; '.join(v.get('issues', []))[:200]}")
        if include_clips and sc.is_hero:
            raw = run_dir / "clips" / f"scene_{sc.id:02d}_hero_raw.mp4"
            if not raw.exists():
                printer(f"⚠️  scene {sc.id}: hero clip not rendered yet")
                continue
            cv = _review_clip(p, sc, state, raw, run_dir)
            reviews += 1
            if _verdict_ok(cv):
                printer(f"✅ scene {sc.id}: animated clip matches")
            else:
                all_ok = False
                bad_clips += 1
                printer(f"❌ scene {sc.id}: CLIP mismatch\n"
                        f"     observed: {cv.get('observed_action', '?')[:160]}\n"
                        f"     issues:   {'; '.join(cv.get('issues', []))[:200]}")
    printer(f"\naudit cost: {reviews} vision reviews (~cents). "
            "Nothing was regenerated.")
    if all_ok:
        printer("✅ every scene matches the story exactly")
    else:
        est = bad_images * IMAGE_COST_USD + bad_clips * CLIP_COST_USD
        printer(f"❌ {bad_images} image(s) + {bad_clips} clip(s) flagged. "
                f"Targeted repair would cost ~${est:.2f} — LOOK at the "
                "flagged files yourself before paying (reviewers have "
                "false-positived on expression/phase nits): delete only the "
                "confirmed-bad scene_XX.png / its clips, fix the scene's "
                "image_prompt in state.json if needed, then `resume`.")
    return all_ok
