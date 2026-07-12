# CLAUDE.md

Automated kids' bedtime-video pipeline: story text in → `final.mp4` + Shorts
teaser + thumbnail + YouTube metadata out. Primary lane: calm bedtime stories,
ages 4–8. Publishing is deliberately manual (stop before upload; videos must
be marked "Made for Kids" / COPPA when uploaded).

## Commands
(dev containers: python deps live in `.venv/` — use `.venv/bin/python`; keys
load with `set -a; . ./.env; set +a`; never read or print `.env` contents)
- Test (offline, always run before committing): `PYTHONPATH=src python -m pytest tests/ -q`
- Mock run (free, no keys): `PYTHONPATH=src python -m kids_story_pipeline run --story examples/sample_story.txt --mock`
- Env check: `PYTHONPATH=src python -m kids_story_pipeline doctor`
- Key smoke test (REAL paid calls, one tiny call per provider):
  `PYTHONPATH=src python -m kids_story_pipeline smoke [--only llm,image,tts,video]`
- Produce a video (REAL, ~$8-15, 15-40 min): `PYTHONPATH=src python -m kids_story_pipeline run --story my_story.txt`
- Resume a paused/failed run: `PYTHONPATH=src python -m kids_story_pipeline resume <RUN_ID> --approve`
  (gate pauses: inspect `runs/<id>/PENDING_APPROVAL.txt` + state.json notes BEFORE approving)

## Architecture
- `src/kids_story_pipeline/nodes.py` — 14 linear nodes (NODES list at bottom is the graph)
- `graph.py` — checkpointed runner; every node emits confidence in [0,1]; gated
  nodes pause the run below `gate_threshold` (adaptive human-in-the-loop)
- `providers/` — seam between mock (deterministic, offline) and real clients
  (Anthropic `claude-sonnet-4-6`, Gemini image "Nano Banana Pro", Kling via
  fal.ai, ElevenLabs voice/SFX/music)
- `gates.py` — deterministic kid-safety checks (Flesch-Kincaid ≤ 3.2,
  scary-word scan, hook length); these are the floor, never skipped
- `config/profiles/bedtime.yaml` — channel identity (pacing, anchors, voices)
- Audio drives visuals: scene clip duration = that scene's narration duration

## Conventions
- Mock-first: all tests must stay offline; ffmpeg paths are real in mock mode
- Never weaken the audience gate or scary-word list without asking the user
- Style/character anchors must be appended to every image prompt (consistency)
- Ask clarifying questions when a task is ambiguous; always test before commit
- Never re-pay for landed assets: nodes skip provider calls when the artifact
  already exists (hero raws, thumb base); resumes reuse everything cached
- A/V sync invariant: scene clips are padded by one `crossfade_s` (hero tails
  by their inner fade) so video length == narration + outro exactly — keep
  this when touching animate/assemble, and keep the e2e duration assertion
- When a gate pauses a real run, review state.json notes before approving —
  every pause so far has flagged a real bug or content issue, not noise

## Status & pending
Production-ready: first video ("The Littlest Star", run 20260705-221817)
approved by the user 2026-07-11 after finishing fixes (A/V sync, smooth
zoom, readable outro/thumbnail text). Publishing stays manual (COPPA).

1. Vision QC reviews every scene image against its lines (action/lighting/
   anatomy) and re-rolls once — but rendered CLIPS are not vision-reviewed
   after animation; Kling can still invent wrong motion. Next gap.
2. v2 roadmap: per-scene SFX buses, read-along captions, 40-min compilation
   builder, LangGraph migration when branching lands
