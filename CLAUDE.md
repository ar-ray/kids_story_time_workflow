# CLAUDE.md

Automated kids' bedtime-video pipeline: story text in → `final.mp4` + Shorts
teaser + thumbnail + YouTube metadata out. Primary lane: calm bedtime stories,
ages 4–8. Publishing is deliberately manual (stop before upload; videos must
be marked "Made for Kids" / COPPA when uploaded).

## Commands
- Test (offline, always run before committing): `PYTHONPATH=src python -m pytest tests/ -q`
- Mock run (free, no keys): `PYTHONPATH=src python -m kids_story_pipeline run --story examples/sample_story.txt --mock`
- Env check: `PYTHONPATH=src python -m kids_story_pipeline doctor`
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
1. `providers/real.py` is UNTESTED against live APIs — verify endpoints/model
   ids against current docs before first paid run (links in README), then add
   a `--smoke` mode that makes one tiny real call per provider
2. Set two ElevenLabs voice ids in `config/profiles/bedtime.yaml`
3. v2 roadmap: vision-LLM QC with auto re-roll (hard-reject uncanny faces),
   per-scene SFX buses, read-along captions, 40-min compilation builder,
   LangGraph migration when branching lands
