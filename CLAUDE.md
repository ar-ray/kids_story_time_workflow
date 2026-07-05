# CLAUDE.md

Automated kids' bedtime-video pipeline: story text in → `final.mp4` + Shorts
teaser + thumbnail + YouTube metadata out. Primary lane: calm bedtime stories,
ages 4–8. Publishing is deliberately manual (stop before upload; videos must
be marked "Made for Kids" / COPPA when uploaded).

## Commands
- Test (offline, always run before committing): `PYTHONPATH=src python -m pytest tests/ -q`
- Mock run (free, no keys): `PYTHONPATH=src python -m kids_story_pipeline run --story examples/sample_story.txt --mock`
- Env check: `PYTHONPATH=src python -m kids_story_pipeline doctor`
- Key smoke test (REAL paid calls, one tiny call per provider):
  `PYTHONPATH=src python -m kids_story_pipeline smoke [--only llm,image,tts,video]`
- Resume a paused run: `PYTHONPATH=src python -m kids_story_pipeline resume <RUN_ID> --approve`

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

## Pending / known gaps (good first tasks)
1. Gemini image gen needs billing enabled on the Google AI Studio project —
   the API free tier has zero image quota (`limit: 0`), so the image/video
   smoke checks stay red until then (llm+tts verified live 2026-07-05)
3. v2 roadmap: vision-LLM QC with auto re-roll (hard-reject uncanny faces),
   per-scene SFX buses, read-along captions, 40-min compilation builder,
   LangGraph migration when branching lands
