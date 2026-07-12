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

## Setup (once per machine)

```bash
# Debian/Ubuntu (incl. dev containers): system deps + a venv (PEP 668)
sudo apt-get install -y python3-pip python3-venv ffmpeg
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Use `.venv/bin/python` wherever the commands below say `python` (or activate
the venv). ffmpeg + ffprobe must be on PATH.

## Quick start (mock mode — free, offline, no keys)

```bash
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

## Producing a video (the ongoing recipe)

One-time: `cp .env.example .env`, fill in the four keys, then confirm they
work with one tiny paid call per provider (`smoke`, below). Per story:

```bash
# 1. write the story as a plain .txt (30+ words; calm, kind, ages 4-8 —
#    scary words and hard vocabulary will trip the audience gate)
export $(grep -v '^#' .env | xargs)
PYTHONPATH=src python -m kids_story_pipeline doctor        # keys + ffmpeg ok?

# 2. run it (~$8-15, 15-40 min; hero animation dominates both)
PYTHONPATH=src python -m kids_story_pipeline run --story my_story.txt

# 3. if a gate pauses the run, read runs/<RUN_ID>/PENDING_APPROVAL.txt and
#    state.json (notes explain exactly what scored low), then either fix the
#    story and re-run, or approve and continue:
PYTHONPATH=src python -m kids_story_pipeline resume <RUN_ID> --approve

# 4. watch runs/<RUN_ID>/final.mp4 yourself — no gate checks visual taste yet
# 5. upload manually: final.mp4 + thumbnail.png, title/description/tags from
#    metadata.json, and ALWAYS mark it "Made for Kids" (COPPA)
```

Runs are checkpointed per node: a crash or `resume` never re-pays for
assets that already landed (images, hero clips, narration, music are all
reused from `runs/<RUN_ID>/`).

| Service | Role | Key |
|---|---|---|
| Anthropic API (`claude-sonnet-4-6`) | script / scenes / personas / packaging | `ANTHROPIC_API_KEY` |
| Gemini image model ("Nano Banana Pro") | character ref + scene art (character consistency) | `GEMINI_API_KEY` |
| Kling image-to-video via fal.ai | hero-scene animation | `FAL_KEY` |
| ElevenLabs | narration (multi-voice), SFX, music (monetization-safe licensing) | `ELEVENLABS_API_KEY` |

**Key check (`smoke`)** — one tiny paid call per provider, run it after
setting keys and whenever a provider errors (live APIs are not exercised by
CI; tests are offline by design). First production video shipped 2026-07.

```bash
PYTHONPATH=src python -m kids_story_pipeline smoke              # all four
PYTHONPATH=src python -m kids_story_pipeline smoke --only llm,tts  # subset
```

(~cents for llm/tts, ~$0.15 for the image, tens of cents for the 3 s video
clip; artifacts land in `runs/smoke-<timestamp>/`.)

Account gotchas learned live: Gemini image generation has **no free-tier
quota** — the key must belong to a billing-enabled Google AI Studio project.
fal.ai locks the account when the balance hits zero — keep it topped up.
ElevenLabs free plans can use **premade** voices via the API but not
*library* voices (402 `paid_plan_required`); `bedtime.yaml` ships with
premade George (narrator) / Matilda (conductor). If providers churn again:
Anthropic <https://docs.claude.com/en/api/overview> ·
Gemini images <https://ai.google.dev/gemini-api/docs/image-generation> ·
ElevenLabs <https://elevenlabs.io/docs/api-reference> ·
fal/Kling <https://fal.ai/models>.

Typical real-mode cost: **~$8–15/video** (hero animation dominates), plus the
ElevenLabs subscription.

## Configuration

Profiles live in `config/profiles/`. `bedtime.yaml` (default) encodes the
channel identity: 105 wpm pacing, long crossfades, 20 s music-only outro,
reading-grade ceiling 3.2, scary-word list, style/character anchors reused in
every image prompt. `adventure.yaml` is the stubbed ages 8–12 mode.

Knobs you'll actually touch:
- `voices:` — ElevenLabs voice ids for narrator/conductor (auditions:
  `runs/voice_auditions/`)
- `tts_speed:` — narration pace (0.85 default; lower = sleepier; slower
  narration automatically lengthens scenes, since audio drives visuals)
- `hero_scene_count:` — Kling animations per video; the main cost lever
- `outro_s`, `crossfade_s`, `music_gain` — feel of the assembly

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

41 tests, all offline: gate logic, xfade/A-V-sync duration math, mux stream
integrity, gate pause→approve→resume, `--no-gate`, state round-trip, scene
chunking + idempotent re-runs, image/hero-clip caching, story-derived
character anchor, vision-QC re-roll + clip invalidation (fake reviewer
verdicts), scene motion prompts, delivery naming, the real-provider request
schemas (monkeypatched `requests`), the smoke runner, and a full end-to-end
mock run asserting all four deliverables.

**Content-fidelity strategy (three layers):** (1) generation prompts demand
the exact physical action, lighting and speaker per scene; (2) at runtime,
`qc_visuals` sends every image + its narration lines to a vision LLM and
re-rolls mismatches once with the reviewer's corrected prompt (stale clips
invalidated automatically) — scoring below the gate threshold still pauses
for a human; (3) the human watch-through before publishing stays the final
check. Offline tests exercise layer 2's mechanics with fake verdicts; they
can't judge real images — that's what the runtime reviewer and you are for.

## Design notes

- **Orchestration:** the v1 graph is strictly linear, so the runner
  (`graph.py`) is a dependency-free checkpointed executor with
  LangGraph-compatible node contracts. When v2 adds branching (QC re-roll
  loops back into image_gen), swap `graph.py` for a LangGraph `StateGraph`.
- **Audio drives visuals:** each scene's clip length is derived from its
  rendered narration duration, so pacing always fits the voice.
- **A/V sync through crossfades:** xfade overlaps clips by `crossfade_s`, so
  every scene clip is padded by exactly one crossfade (hero tails by their
  inner fade too). Video length == narration + outro, and each scene's
  visuals start on its first narrated line.
- **Smooth Ken Burns:** zoompan input is supersampled to 4× output width —
  at 1× the crop window quantizes to whole pixels and slow zooms shake.

## Roadmap (v2)

- ~~Vision-LLM QC on generated images with auto re-roll~~ — shipped: every
  scene image is reviewed against its narration lines (action, lighting,
  anatomy) and re-rolled once on mismatch. Clips are not yet vision-reviewed
  post-animation — that's the remaining piece
- Per-scene SFX buses (single ambience bed today)
- Read-along captions from ElevenLabs word timestamps (adventure mode)
- Compilation builder (3–4 stories → 40-min sleep video)
- LangGraph migration once branching lands
