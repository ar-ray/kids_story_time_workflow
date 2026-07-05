# kids_story_time_workflow

Automated pipeline that turns a children's story (text file in → finished
video out) into a calm, Milo-style bedtime video for **ages 4–8**: narrated
script, consistent-character scene art, subtle animation on hero scenes,
music bed, ffmpeg assembly, Shorts teaser, thumbnail, and YouTube metadata.

Deliberately stops **before upload** — you get `final.mp4` + `metadata.json`
and publish manually (remember to mark uploads **Made for Kids** / COPPA).

## Pipeline

```
story.txt ─► intake ─► script_agent ─► scene_director ─► audience_gate*
        ─► character_ref ─► image_gen ─► qc_visuals* ─► voice ─► animate
        ─► music ─► assemble* ─► shorts ─► thumbnail ─► package_meta
```

`*` = gated nodes. Every node emits a **confidence score**; the run is fully
hands-off unless a gated node scores below `gate_threshold` (0.75), in which
case the run **pauses**, writes `runs/<id>/PENDING_APPROVAL.txt`, and waits
for `resume <id> --approve`. That's the adaptive human-in-the-loop: zero
touches on good runs, one tap on shaky ones. `--no-gate` disables pausing.

The audience gate combines deterministic checks that always run (Flesch-
Kincaid reading level vs. the 4–8 ceiling, scary-word scan, hook length) with
an LLM kid+parent persona review in real mode.

## Quick start (mock mode — free, offline, no keys)

```bash
pip install -r requirements.txt          # needs ffmpeg + ffprobe on PATH
PYTHONPATH=src python -m kids_story_pipeline run \
    --story examples/sample_story.txt --mock
```

Outputs land in `runs/<run_id>/`: `final.mp4`, `shorts_teaser.mp4`,
`thumbnail.png`, `metadata.json`, plus per-scene assets and `state.json`
(the checkpoint — every node is resumable).

Mock mode swaps only the paid APIs (LLM, images, video, audio) for
deterministic local fixtures; **orchestration, timing math, and ffmpeg
rendering are the real code paths**, so a green mock run means the pipeline
itself works.

## Real mode

```bash
cp .env.example .env    # fill in keys
export $(grep -v '^#' .env | xargs)
PYTHONPATH=src python -m kids_story_pipeline doctor    # sanity-check
PYTHONPATH=src python -m kids_story_pipeline run --story my_story.txt
```

| Service | Role | Key |
|---|---|---|
| Anthropic API (`claude-sonnet-4-6`) | script / scenes / personas / packaging | `ANTHROPIC_API_KEY` |
| Gemini image model ("Nano Banana Pro") | character ref + scene art (character consistency) | `GEMINI_API_KEY` |
| Kling image-to-video via fal.ai | hero-scene animation | `FAL_KEY` |
| ElevenLabs | narration (multi-voice), SFX, music (monetization-safe licensing) | `ELEVENLABS_API_KEY` |

⚠️ **Before your first paid run:** the real clients in
`src/kids_story_pipeline/providers/real.py` are written against provider docs
as of mid-2026 but are **not exercised by CI** (tests are offline by design).
Verify model ids/endpoints — they churn:
Anthropic <https://docs.claude.com/en/api/overview> ·
Gemini images <https://ai.google.dev/gemini-api/docs/image-generation> ·
ElevenLabs <https://elevenlabs.io/docs/api-reference> ·
fal/Kling <https://fal.ai/models>. Also set the two ElevenLabs voice ids in
`config/profiles/bedtime.yaml`.

Typical real-mode cost: **~$8–15/video** (hero animation dominates), plus the
ElevenLabs subscription.

## Configuration

Profiles live in `config/profiles/`. `bedtime.yaml` (default) encodes the
channel identity: 105 wpm pacing, long crossfades, 20 s music-only outro,
reading-grade ceiling 3.2, scary-word list, style/character anchors reused in
every image prompt. `adventure.yaml` is the stubbed ages 8–12 mode.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

10 tests: gate logic, xfade duration math, mux stream integrity, gate
pause→approve→resume, hands-off `--no-gate` behavior, state round-trip, and a
full end-to-end mock run asserting all four deliverables. All offline.

## Design notes

- **Orchestration:** the v1 graph is strictly linear, so the runner
  (`graph.py`) is a dependency-free checkpointed executor with
  LangGraph-compatible node contracts. When v2 adds branching (QC re-roll
  loops back into image_gen), swap `graph.py` for a LangGraph `StateGraph`.
- **Audio drives visuals:** each scene's clip length is derived from its
  rendered narration duration, so pacing always fits the voice.

## Roadmap (v2)

- Vision-LLM QC on generated images/clips with auto re-roll (hard-reject
  uncanny faces — critical for kids' content)
- Per-scene SFX buses (single ambience bed today)
- Read-along captions from ElevenLabs word timestamps (adventure mode)
- Compilation builder (3–4 stories → 40-min sleep video)
- LangGraph migration once branching lands
