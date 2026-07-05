"""Pipeline nodes. Each node: fn(state, providers, profile, run_dir) -> confidence.

Nodes mutate state and return a confidence score in [0, 1]. The graph runner
checkpoints after every node and pauses for human approval when a node's
confidence falls below the profile threshold (adaptive human-in-the-loop).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from . import ffmpeg_utils as ff
from . import gates
from .config import Profile
from .providers import Providers
from .state import Line, PipelineState, Scene


# --------------------------------------------------------------------------- 
def intake(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    if len(state.story_text.split()) < 30:
        state.notes.append("intake: story very short (<30 words)")
        return 0.5
    state.style_anchor = prof.raw.get(
        "style_anchor",
        "storybook illustration, soft watercolor, deep blue and warm gold night "
        "palette, cozy dreamy atmosphere, children's book art")
    state.character_anchor = prof.raw.get(
        "character_anchor",
        "a young child with curly dark hair in mustard-yellow pajamas")
    return 1.0


def script_agent(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    system = ("SCRIPT_TASK: You adapt children's stories into bedtime narration "
              f"scripts for ages {prof.target_age}. Short sentences, a soothing "
              "repeated refrain, gentle pacing. Tag dialogue lines with the "
              "speaking character's role. Return JSON: "
              '{"title": str, "refrain": str, "lines": [{"text": str, "role": str}]}')
    result = p.llm.complete_json(system, "STORY:\n" + state.story_text)
    state.title = result.get("title", "A Sleepy Story")
    state.refrain = result.get("refrain", "")
    lines = [Line(**l) for l in result.get("lines", []) if l.get("text")]
    if not lines:
        return 0.0
    # stash on a single provisional scene; scene_director re-buckets
    state.scenes = [Scene(id=0, title="__all__", lines=lines)]
    return 1.0


def scene_director(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    all_lines = state.scenes[0].lines
    numbered = "\n".join(f"{i}: {l.text}" for i, l in enumerate(all_lines))
    system = ("SCENE_TASK: Split the numbered narration lines into 6-16 visual "
              "scenes for a bedtime video. Return JSON: {\"scenes\": [{\"title\": str, "
              "\"line_start\": int, \"line_end\": int, \"image_prompt\": str, "
              "\"sfx_prompt\": str}]} with contiguous, non-overlapping ranges.")
    result = p.llm.complete_json(system, numbered)
    scenes: list[Scene] = []
    for i, s in enumerate(result.get("scenes", [])):
        chunk = all_lines[s["line_start"]:s["line_end"]]
        if not chunk:
            continue
        scenes.append(Scene(
            id=i, title=s.get("title", f"Scene {i + 1}"), lines=chunk,
            image_prompt=f"{s.get('image_prompt', '')}, {state.style_anchor}",
            sfx_prompt=s.get("sfx_prompt", "soft night ambience"),
        ))
    if not scenes:
        return 0.0
    # choose hero scenes evenly across the video
    n_hero = min(prof.hero_scene_count, len(scenes))
    if n_hero:
        step = max(1, len(scenes) // n_hero)
        for idx in range(0, len(scenes), step):
            if sum(sc.is_hero for sc in scenes) < n_hero:
                scenes[idx].is_hero = True
    state.scenes = scenes
    covered = sum(len(sc.lines) for sc in scenes)
    return round(covered / len(all_lines), 3)


def audience_gate(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    """Kid + parent review: deterministic floor checks, then persona LLM."""
    full_text = " ".join(sc.narration_text for sc in state.scenes)
    first = state.scenes[0].narration_text if state.scenes else ""
    checks = [
        gates.reading_level_gate(full_text, prof.max_reading_grade),
        gates.scary_content_gate(full_text, prof.scary_words),
        gates.hook_gate(first),
    ]
    persona = p.llm.complete_json(
        "PERSONA_TASK: Review this bedtime script as (a) a child aged "
        f"{prof.target_age} and (b) their parent. Return JSON: "
        '{"kid_score": 0..1, "parent_score": 0..1, "comments": [str]}',
        full_text,
    )
    for r in checks:
        state.notes.append(f"gate/{r.name}: conf={r.confidence} ({r.details})")
    for c in persona.get("comments", []):
        state.notes.append(f"gate/persona: {c}")
    hard = gates.combine(checks)
    soft = min(float(persona.get("kid_score", 1.0)),
               float(persona.get("parent_score", 1.0)))
    return round(min(hard, soft), 3)


def character_ref(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    out = run_dir / "assets" / "character_ref.png"
    prompt = (f"character reference sheet, front and side view, {state.character_anchor}, "
              f"{state.style_anchor}, neutral background")
    p.images.generate(prompt, out, size=prof.size)
    state.character_ref_path = str(out)
    return 1.0 if out.exists() and out.stat().st_size > 0 else 0.0


def image_gen(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    ref = Path(state.character_ref_path) if state.character_ref_path else None
    ok = 0
    for sc in state.scenes:
        out = run_dir / "assets" / f"scene_{sc.id:02d}.png"
        p.images.generate(sc.image_prompt, out, reference=ref, size=prof.size)
        if out.exists() and out.stat().st_size > 0:
            sc.image_path = str(out)
            ok += 1
    return round(ok / len(state.scenes), 3)


def qc_visuals(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    """Mock: structural checks. Real mode: replace with a vision-LLM review
    that hard-rejects uncanny faces / distorted figures (see README roadmap)."""
    bad = [sc.id for sc in state.scenes
           if not sc.image_path or not Path(sc.image_path).exists()]
    if bad:
        state.notes.append(f"qc_visuals: missing images for scenes {bad}")
        return 0.0
    with Image.open(state.scenes[0].image_path) as im:
        if im.size != prof.size:
            state.notes.append(f"qc_visuals: unexpected size {im.size}")
            return 0.6
    return 0.9


def voice(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    audio_dir = run_dir / "audio"
    for sc in state.scenes:
        line_files: list[Path] = []
        for j, line in enumerate(sc.lines):
            f = audio_dir / f"s{sc.id:02d}_l{j:02d}.wav"
            p.audio.tts(line.text, line.role, f)
            line_files.append(f)
        raw = audio_dir / f"scene_{sc.id:02d}_raw.wav"
        ff.concat_audio(line_files, raw) if len(line_files) > 1 else raw.write_bytes(line_files[0].read_bytes())
        padded = audio_dir / f"scene_{sc.id:02d}.wav"
        ff.pad_audio(raw, padded, prof.scene_pause_s)
        sc.audio_path = str(padded)
        sc.audio_duration_s = round(ff.probe_duration(padded), 3)
    return 1.0


def animate(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    """Visual clip per scene, sized to that scene's narration audio."""
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for sc in state.scenes:
        dur = max(2.0, sc.audio_duration_s)
        out = clips_dir / f"scene_{sc.id:02d}.mp4"
        if sc.is_hero:
            raw = clips_dir / f"scene_{sc.id:02d}_hero_raw.mp4"
            p.video.animate(Path(sc.image_path), f"gentle slow motion, {sc.title}",
                            dur, raw)
            ff.normalize_clip(raw, out, size=prof.size, fps=prof.fps)
            # hero models cap at short clips; hold the last look via kenburns if short
            if ff.probe_duration(out) < dur - 0.5:
                tail = clips_dir / f"scene_{sc.id:02d}_tail.mp4"
                ff.make_kenburns_clip(Path(sc.image_path),
                                      dur - ff.probe_duration(out), tail,
                                      size=prof.size, fps=prof.fps)
                joined = clips_dir / f"scene_{sc.id:02d}_joined.mp4"
                ff.xfade_concat([out, tail], joined, fade_s=min(0.8, prof.crossfade_s))
                out = joined
        else:
            ff.make_kenburns_clip(Path(sc.image_path), dur, out,
                                  size=prof.size, fps=prof.fps)
        sc.clip_path = str(out)
    return 1.0


def music(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    total = sum(sc.audio_duration_s for sc in state.scenes) + prof.outro_s
    out = run_dir / "audio" / "music.wav"
    p.audio.music("gentle music-box lullaby, slow, warm, sparse piano, no drums",
                  total + 2.0, out)
    state.music_path = str(out)
    return 1.0


def assemble(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    # outro card
    card = run_dir / "assets" / "outro.png"
    img = Image.new("RGB", prof.size, (18, 22, 44))
    d = ImageDraw.Draw(img)
    d.text((prof.width // 2 - 80, prof.height // 2), "goodnight",
           fill=(200, 200, 215))
    img.save(card)
    outro_clip = run_dir / "clips" / "outro.mp4"
    ff.make_kenburns_clip(card, prof.outro_s, outro_clip,
                          size=prof.size, fps=prof.fps, zoom_rate=0.0002)

    clips = [Path(sc.clip_path) for sc in state.scenes] + [outro_clip]
    video = run_dir / "video_silent.mp4"
    ff.xfade_concat(clips, video, fade_s=prof.crossfade_s)

    narr_files = [Path(sc.audio_path) for sc in state.scenes]
    outro_sil = run_dir / "audio" / "outro_silence.wav"
    ff.synth_silence(outro_sil, prof.outro_s)
    narration = run_dir / "audio" / "narration_full.wav"
    ff.concat_audio(narr_files + [outro_sil], narration)

    mixed = run_dir / "audio" / "final_mix.wav"
    ff.mix_music(narration, Path(state.music_path), mixed,
                 music_gain=prof.music_gain)

    master = run_dir / "final.mp4"
    ff.mux(video, mixed, master)
    state.master_video_path = str(master)

    streams = ff.probe_streams(master)
    ok = "video" in streams and "audio" in streams
    return 1.0 if ok else 0.0


def shorts(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    out = run_dir / "shorts_teaser.mp4"
    ff.shorts_cutdown(Path(state.master_video_path), out, max_s=prof.shorts_max_s)
    state.shorts_path = str(out)
    return 1.0


def thumbnail(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    base = run_dir / "assets" / "thumb_base.png"
    p.images.generate(
        f"{state.character_anchor}, cozy magical night scene, {state.style_anchor}, "
        "thumbnail composition, large clear subject", base, size=prof.size)
    out = run_dir / "thumbnail.png"
    with Image.open(base) as im:
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        label = f"{state.title}  ({'~' + str(int(_total_minutes(state))) + ' min'})"
        d.rectangle([0, prof.height - 90, prof.width, prof.height],
                    fill=(18, 22, 44))
        d.text((30, prof.height - 65), label[:70], fill=(240, 236, 210))
        im.save(out)
    state.thumbnail_path = str(out)
    return 1.0


def package_meta(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    result = p.llm.complete_json(
        "PACKAGE_TASK: Write a warm 2-3 sentence YouTube description and 5-8 "
        'tags for a kids bedtime story video. JSON: {"description": str, "tags": [str]}',
        f"Title: {state.title}\nRefrain: {state.refrain}")
    minutes = int(_total_minutes(state))
    meta = {
        "title": f"{state.title} 🌙 Calming Bedtime Story for Kids ({minutes} min)",
        "description": result.get("description", "") +
            "\n\n🌙 A calming sleep story for ages " + prof.target_age +
            ". Music continues after the story ends.\n\n" +
            "⚠️ Remember to mark this video 'Made for Kids' (COPPA) when uploading.",
        "tags": result.get("tags", []),
        "made_for_kids": True,
        "runtime_minutes": minutes,
    }
    out = run_dir / "metadata.json"
    out.write_text(json.dumps(meta, indent=2))
    state.metadata_path = str(out)
    return 1.0


def _total_minutes(state: PipelineState) -> float:
    return sum(sc.audio_duration_s for sc in state.scenes) / 60.0


# Ordered graph. gated=True nodes can pause the run on low confidence.
NODES: list[tuple[str, object, bool]] = [
    ("intake", intake, False),
    ("script_agent", script_agent, False),
    ("scene_director", scene_director, True),
    ("audience_gate", audience_gate, True),
    ("character_ref", character_ref, False),
    ("image_gen", image_gen, True),
    ("qc_visuals", qc_visuals, True),
    ("voice", voice, False),
    ("animate", animate, False),
    ("music", music, False),
    ("assemble", assemble, True),
    ("shorts", shorts, False),
    ("thumbnail", thumbnail, False),
    ("package_meta", package_meta, False),
]
