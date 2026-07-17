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
              f"scripts for ages {prof.target_age}. Every sentence must be very "
              "short: aim for 5-9 words, never more than 12. Simple everyday "
              f"words. Target US reading grade {prof.max_reading_grade} or "
              "below (Flesch-Kincaid) — this is checked by an automated gate. "
              "Use a soothing repeated refrain and gentle pacing. Tag dialogue "
              "lines with the speaking character's role. Also describe the "
              "story's main character visually in character_anchor (species, "
              "colors, distinctive features — it anchors every illustration). "
              "Return JSON: "
              '{"title": str, "refrain": str, "character_anchor": str, '
              '"lines": [{"text": str, "role": str}]}')
    result = p.llm.complete_json(system, "STORY:\n" + state.story_text)
    state.title = result.get("title", "A Sleepy Story")
    state.refrain = result.get("refrain", "")
    # the story's own main character beats the profile's generic anchor
    derived = (result.get("character_anchor") or "").strip()
    if derived:
        state.character_anchor = derived
    lines = [Line(**l) for l in result.get("lines", []) if l.get("text")]
    if not lines:
        return 0.0
    # stash on a single provisional scene; scene_director re-buckets
    state.scenes = [Scene(id=0, title="__all__", lines=lines)]
    return 1.0


def scene_director(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    # Flatten across ALL scenes, not scenes[0]: a re-run after a pause/crash
    # sees its own previous bucketing (this node overwrites state.scenes),
    # and reading only scenes[0] would shrink the script to one bucket.
    all_lines = [l for sc in state.scenes for l in sc.lines]
    numbered = "\n".join(f"{i}: {l.text}" for i, l in enumerate(all_lines))
    system = ("SCENE_TASK: Split the numbered narration lines into 6-16 visual "
              "scenes for a bedtime video. Return JSON: {\"scenes\": [{\"title\": str, "
              "\"line_start\": int, \"line_end\": int, \"image_prompt\": str, "
              "\"motion_prompt\": str, \"sfx_prompt\": str}]} with contiguous, "
              "non-overlapping ranges (line_end is EXCLUSIVE, python-slice "
              "style). Each image_prompt must faithfully match the lighting, "
              "time of day and mood the story describes in those lines — a "
              "dark room stays dark, lit only by light sources the story "
              "mentions. Depict the EXACT physical action in those lines: "
              "state explicitly who touches what and how (e.g. 'hanging by "
              "his MOUTH biting the middle of the stick, flippers dangling "
              "free' — never holding with hands/paws if the story says "
              "biting). Give dialogue exchanges their own scene, with the "
              "speaking characters visible. motion_prompt is one short "
              "sentence of subtle animation faithful to that same action.")
    result = p.llm.complete_json(system, numbered)
    # Chunk by consecutive line_starts only: models disagree on whether
    # line_end is inclusive or exclusive, and trusting it silently drops the
    # last line of every scene. Starts alone guarantee full coverage.
    raw = sorted((s for s in result.get("scenes", [])
                  if isinstance(s.get("line_start"), int)),
                 key=lambda s: s["line_start"])
    starts = [min(max(0, s["line_start"]), len(all_lines)) for s in raw]
    if starts:
        starts[0] = 0  # never drop the opening lines
    ends = starts[1:] + [len(all_lines)]
    scenes: list[Scene] = []
    for i, (s, start, end) in enumerate(zip(raw, starts, ends)):
        chunk = all_lines[start:end]
        if not chunk:
            continue
        scenes.append(Scene(
            id=i, title=s.get("title", f"Scene {i + 1}"), lines=chunk,
            image_prompt=f"{s.get('image_prompt', '')}, {state.style_anchor}",
            motion_prompt=s.get("motion_prompt", ""),
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
    ok = generated = reused = 0
    for sc in state.scenes:
        out = run_dir / "assets" / f"scene_{sc.id:02d}.png"
        # paid asset: only generate scenes whose image hasn't landed yet
        # (resume after crash, or targeted re-rolls that deleted the file)
        if not out.exists() or out.stat().st_size == 0:
            p.images.generate(sc.image_prompt, out, reference=ref, size=prof.size)
            generated += 1
        else:
            reused += 1
        if out.exists() and out.stat().st_size > 0:
            sc.image_path = str(out)
            ok += 1
    state.notes.append(f"image_gen: cost — generated {generated}, "
                       f"reused {reused}")
    return round(ok / len(state.scenes), 3)


MAX_IMAGE_REROLLS = 1


def qc_visuals(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    """Structural checks always; in real mode also a vision-LLM review of
    every scene image against its narration lines, with one corrective
    re-roll for images that don't depict the described action/setting."""
    bad = [sc.id for sc in state.scenes
           if not sc.image_path or not Path(sc.image_path).exists()]
    if bad:
        state.notes.append(f"qc_visuals: missing images for scenes {bad}")
        return 0.0
    with Image.open(state.scenes[0].image_path) as im:
        if im.size != prof.size:
            state.notes.append(f"qc_visuals: unexpected size {im.size}")
            return 0.6
    if state.mock:
        return 0.9
    return min(0.9, _vision_review(state, p, prof, run_dir))


def _vision_review(state: PipelineState, p: Providers, prof: Profile,
                   run_dir: Path) -> float:
    """Check each image tells its lines' story; re-roll mismatches once."""
    ref = Path(state.character_ref_path) if state.character_ref_path else None
    passed = rerolls = 0
    for sc in state.scenes:
        verdict = {}
        for attempt in range(MAX_IMAGE_REROLLS + 1):
            verdict = _review_one(p, sc, state)
            if _verdict_ok(verdict):
                break
            issues = "; ".join(verdict.get("issues", []))[:300]
            state.notes.append(
                f"qc_visuals: scene {sc.id} mismatch "
                f"(observed: {verdict.get('observed_action', '?')[:120]}): "
                f"{issues}")
            if attempt >= MAX_IMAGE_REROLLS:
                break
            corrected = (verdict.get("corrected_prompt") or "").strip()
            if corrected:
                sc.image_prompt = f"{corrected}, {state.style_anchor}"
            p.images.generate(sc.image_prompt, Path(sc.image_path),
                              reference=ref, size=prof.size)
            _invalidate_clip(sc, run_dir)
            rerolls += 1
            state.notes.append(f"qc_visuals: scene {sc.id} re-rolled")
        if _verdict_ok(verdict):
            passed += 1
    state.notes.append(
        f"qc_visuals: cost — {len(state.scenes) + rerolls} vision reviews, "
        f"{rerolls} image re-rolls")
    return round(passed / len(state.scenes), 3)


_QC_SYSTEM = (
    "VISION_QC_TASK: You are reviewing imagery for a kids' story video. "
    "FIRST describe, then judge — fill observed_action with what the "
    "image(s) ACTUALLY show for every physical contact between characters "
    "and objects (e.g. 'turtle lies ON TOP of the stick, flippers over it'), "
    "and expected_action with what the lines + story require (continuity: a "
    "mechanism the story establishes anywhere — e.g. the character travels "
    "hanging by its MOUTH from a stick — must hold in EVERY scene that "
    "shows it, even if this scene's lines don't repeat the detail). Set "
    "action_exact=true ONLY if they agree in every detail: on-top vs "
    "hanging-below, mouth vs limbs, which part touches what. Being on/over "
    "an object when the story says hanging beneath it by the mouth is "
    "WRONG. Also check setting/lighting match the lines and there is no "
    "distorted anatomy, extra limbs or uncanny faces. Minor style license "
    "is fine; mechanism, lighting or anatomy errors are not. Return JSON: "
    '{"observed_action": str, "expected_action": str, "action_exact": bool, '
    '"matches": bool, "issues": [str], "corrected_prompt": str} — matches '
    "must be false when action_exact is false; corrected_prompt is a full "
    "replacement image prompt fixing the issues (empty if matches)."
)


def _qc_context(state: PipelineState, sc) -> str:
    return (f"FULL STORY (for continuity):\n{state.story_text[:3000]}\n\n"
            f"MAIN CHARACTER: {state.character_anchor}\n\n"
            f"NARRATION LINES THIS IMAGERY ILLUSTRATES:\n{sc.narration_text}"
            f"\n\nIMAGE PROMPT USED:\n{sc.image_prompt}")


def _verdict_ok(v: dict) -> bool:
    return bool(v.get("matches")) and bool(v.get("action_exact", True))


def _review_one(p: Providers, sc, state: PipelineState) -> dict:
    return p.llm.complete_json(_QC_SYSTEM, _qc_context(state, sc),
                               images=[Path(sc.image_path)])


def _invalidate_clip(sc, run_dir: Path) -> None:
    """A re-rolled image makes this scene's rendered clips stale."""
    clips_dir = run_dir / "clips"
    for stem in (f"scene_{sc.id:02d}", f"scene_{sc.id:02d}_hero_raw",
                 f"scene_{sc.id:02d}_tail", f"scene_{sc.id:02d}_joined"):
        f = clips_dir / f"{stem}.mp4"
        if f.exists():
            f.unlink()
    sc.clip_path = None


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
    """Visual clip per scene, sized to that scene's narration audio.
    Hero clips are vision-reviewed via extracted frames (animation models
    can drift from the source image's action) with a one-re-render cap."""
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rendered = reused = clip_rerolls = failed = 0
    for sc in state.scenes:
        # pad every scene clip by one crossfade: xfade_concat overlaps clips
        # by fade_s, so without padding the video track ends n_fades*fade_s
        # before the narration and every scene drifts ahead of its audio
        dur = max(2.0, sc.audio_duration_s) + prof.crossfade_s
        out = clips_dir / f"scene_{sc.id:02d}.mp4"
        if sc.is_hero:
            raw = clips_dir / f"scene_{sc.id:02d}_hero_raw.mp4"
            for attempt in range(2):
                # hero renders are the expensive step — never re-pay for a
                # clip a previous (crashed/paused) attempt already downloaded
                if not raw.exists() or raw.stat().st_size == 0:
                    motion = (sc.motion_prompt
                              or f"gentle slow motion, {sc.title}")
                    p.video.animate(Path(sc.image_path), motion, dur, raw)
                    rendered += 1
                elif attempt == 0:
                    reused += 1
                if state.mock or _clip_review_ok(p, sc, state, raw, run_dir):
                    break
                if attempt == 0:  # one paid re-render, with sharpened motion
                    raw.unlink()
                    sc.motion_prompt = ((sc.motion_prompt or "").rstrip("; ")
                        + "; CRITICAL: keep the exact physical contact and "
                          "pose from the source image unchanged throughout")
                    clip_rerolls += 1
                    state.notes.append(f"animate: scene {sc.id} clip re-rolled")
                else:
                    failed += 1
                    state.notes.append(f"animate: scene {sc.id} clip still "
                                       "mismatched after re-roll")
            ff.normalize_clip(raw, out, size=prof.size, fps=prof.fps)
            # hero models cap at short clips; hold the last look via kenburns if short
            if ff.probe_duration(out) < dur - 0.5:
                fade = min(0.8, prof.crossfade_s)
                tail = clips_dir / f"scene_{sc.id:02d}_tail.mp4"
                # + fade: the join overlaps by fade_s, so the tail must be
                # longer by exactly that much for the clip to land on dur
                ff.make_kenburns_clip(Path(sc.image_path),
                                      dur - ff.probe_duration(out) + fade,
                                      tail, size=prof.size, fps=prof.fps)
                joined = clips_dir / f"scene_{sc.id:02d}_joined.mp4"
                ff.xfade_concat([out, tail], joined, fade_s=fade)
                out = joined
        else:
            ff.make_kenburns_clip(Path(sc.image_path), dur, out,
                                  size=prof.size, fps=prof.fps)
        sc.clip_path = str(out)
    state.notes.append(f"animate: cost — hero rendered {rendered}, "
                       f"reused {reused}, clip re-rolls {clip_rerolls}")
    if failed:
        return round(1 - failed / len(state.scenes), 3)
    return 1.0


def _clip_review_ok(p: Providers, sc, state: PipelineState, raw: Path,
                    run_dir: Path) -> bool:
    """Review two extracted frames of a hero clip against the story."""
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    dur = ff.probe_duration(raw)
    frames = []
    for tag, t in (("a", dur * 0.25), ("b", dur * 0.75)):
        frames.append(ff.extract_frame(
            raw, t, qc_dir / f"scene_{sc.id:02d}_{tag}.png"))
    v = p.llm.complete_json(
        _QC_SYSTEM + " The images are FRAMES FROM THE ANIMATED CLIP for "
        "this scene — animation can drift from the approved source image, "
        "so judge the frames themselves.",
        _qc_context(state, sc), images=frames)
    if _verdict_ok(v):
        return True
    state.notes.append(
        f"animate: scene {sc.id} clip mismatch "
        f"(observed: {v.get('observed_action', '?')[:120]}): "
        + "; ".join(v.get("issues", []))[:200])
    return False


def music(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    total = sum(sc.audio_duration_s for sc in state.scenes) + prof.outro_s
    out = run_dir / "audio" / "music.wav"
    p.audio.music("gentle music-box lullaby, slow, warm, sparse piano, no drums",
                  total + 2.0, out)
    state.music_path = str(out)
    return 1.0


def _font(px: int):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def assemble(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    # outro card — large, centered text (PIL's default font is ~11px)
    card = run_dir / "assets" / "outro.png"
    img = Image.new("RGB", prof.size, (18, 22, 44))
    d = ImageDraw.Draw(img)
    font = _font(max(48, prof.height // 8))
    text = "good night"
    box = d.textbbox((0, 0), text, font=font)
    d.text(((prof.width - (box[2] - box[0])) // 2,
            (prof.height - (box[3] - box[1])) // 2),
           text, fill=(230, 224, 200), font=font)
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
    if not base.exists() or base.stat().st_size == 0:  # keep paid asset on re-runs
        p.images.generate(
            f"{state.character_anchor}, cozy magical night scene, {state.style_anchor}, "
            "thumbnail composition, large clear subject", base, size=prof.size)
    out = run_dir / "thumbnail.png"
    with Image.open(base) as im:
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        label = f"{state.title}  ({'~' + str(_total_minutes(state, prof)) + ' min'})"
        band_h = max(90, prof.height // 7)
        font = _font(int(band_h * 0.45))
        d.rectangle([0, prof.height - band_h, prof.width, prof.height],
                    fill=(18, 22, 44))
        box = d.textbbox((0, 0), label[:70], font=font)
        d.text((30, prof.height - band_h + (band_h - (box[3] - box[1])) // 2),
               label[:70], fill=(240, 236, 210), font=font)
        im.save(out)
    state.thumbnail_path = str(out)
    return 1.0


def package_meta(state: PipelineState, p: Providers, prof: Profile, run_dir: Path) -> float:
    result = p.llm.complete_json(
        "PACKAGE_TASK: Write a warm 2-3 sentence YouTube description and 5-8 "
        'tags for a kids bedtime story video. JSON: {"description": str, "tags": [str]}',
        f"Title: {state.title}\nRefrain: {state.refrain}")
    minutes = _total_minutes(state, prof)
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


def _total_minutes(state: PipelineState, prof: Profile) -> int:
    """Rounded length of the final video (narration + outro), min 1."""
    total_s = sum(sc.audio_duration_s for sc in state.scenes) + prof.outro_s
    return max(1, round(total_s / 60.0))


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
