"""Tests. Everything runs offline in mock mode; ffmpeg work is real."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kids_story_pipeline import ffmpeg_utils as ff
from kids_story_pipeline import gates, graph
from kids_story_pipeline.config import load_profile
from kids_story_pipeline.state import PipelineState

SAMPLE = (Path(__file__).parents[1] / "examples" / "sample_story.txt").read_text()

# small/fast render settings for tests
FAST = {"outro_s": 3.0, "crossfade_s": 0.5, "hero_scene_count": 1,
        "shorts_max_s": 8.0, "width": 640, "height": 360, "fps": 12}


@pytest.fixture()
def fast_profile():
    return load_profile("bedtime", overrides=FAST)


# ---- gates -----------------------------------------------------------------

def test_reading_level_simple_text_passes():
    r = gates.reading_level_gate("The cat sat. The dog ran. It was fun.", 3.2)
    assert r.passed and r.confidence == 1.0


def test_reading_level_complex_text_degrades():
    hard = ("Notwithstanding the extraordinarily complicated meteorological "
            "circumstances, the intrepid protagonist persevered relentlessly "
            "throughout innumerable consecutive interminable evenings.")
    r = gates.reading_level_gate(hard, 3.2)
    assert r.confidence < 0.75


def test_scary_content_gate_flags_words():
    r = gates.scary_content_gate("The monster hid in the dark.", ["monster"])
    assert not r.passed and r.confidence <= 0.75
    ok = gates.scary_content_gate("The bunny hid in the soft dark.", ["monster"])
    assert ok.passed and ok.confidence == 1.0


def test_sample_story_is_age_appropriate():
    assert gates.flesch_kincaid_grade(SAMPLE) <= 3.2
    assert gates.scary_content_gate(
        SAMPLE, load_profile("bedtime").scary_words).passed


# ---- ffmpeg math -------------------------------------------------------------

def test_xfade_concat_duration(tmp_path):
    from PIL import Image
    clips = []
    for i, dur in enumerate([3.0, 3.0, 3.0]):
        img = tmp_path / f"i{i}.png"
        Image.new("RGB", (320, 180), (30 * i, 40, 80)).save(img)
        clip = tmp_path / f"c{i}.mp4"
        ff.make_kenburns_clip(img, dur, clip, size=(320, 180), fps=12)
        clips.append(clip)
    out = tmp_path / "joined.mp4"
    ff.xfade_concat(clips, out, fade_s=0.5)
    # expected: sum(d) - (n-1)*fade = 9 - 1.0 = 8.0
    assert ff.probe_duration(out) == pytest.approx(8.0, abs=0.4)


def test_mux_and_streams(tmp_path):
    from PIL import Image
    img = tmp_path / "i.png"
    Image.new("RGB", (320, 180), (20, 30, 60)).save(img)
    vid = tmp_path / "v.mp4"
    ff.make_kenburns_clip(img, 2.0, vid, size=(320, 180), fps=12)
    aud = tmp_path / "a.wav"
    ff.synth_tone(aud, 2.0)
    out = tmp_path / "m.mp4"
    ff.mux(vid, aud, out)
    assert sorted(ff.probe_streams(out)) == ["audio", "video"]


def test_script_agent_derives_story_character_anchor(fast_profile, tmp_path):
    """Images must suit the story: a turtle tale anchors on the turtle the
    LLM describes, not the profile's generic default character."""
    from kids_story_pipeline import nodes

    class AnchorLLM:
        def complete_json(self, system, prompt):
            return {"title": "T", "refrain": "r",
                    "character_anchor": "a small green turtle, round shell",
                    "lines": [{"text": "Hi.", "role": "narrator"}]}

    class NoAnchorLLM(AnchorLLM):
        def complete_json(self, system, prompt):
            d = AnchorLLM.complete_json(self, system, prompt)
            d.pop("character_anchor")
            return d

    class P:
        llm = AnchorLLM()

    state = PipelineState(run_id="ca", story_text="word " * 40,
                          profile_name="bedtime")
    nodes.intake(state, P(), fast_profile, tmp_path)   # profile default
    nodes.script_agent(state, P(), fast_profile, tmp_path)
    assert state.character_anchor == "a small green turtle, round shell"

    state2 = PipelineState(run_id="ca2", story_text="word " * 40,
                           profile_name="bedtime")
    P.llm = NoAnchorLLM()
    nodes.intake(state2, P(), fast_profile, tmp_path)
    nodes.script_agent(state2, P(), fast_profile, tmp_path)
    assert state2.character_anchor  # falls back to the profile anchor


def test_deliver_copies_video_named_after_story(tmp_path):
    from kids_story_pipeline.cli import _deliver
    master = tmp_path / "final.mp4"
    master.write_bytes(b"vid")
    outdir = tmp_path / "storyvideos"
    dest = _deliver(master, outdir, "toby_the_turtle")
    assert dest == outdir / "toby_the_turtle.mp4"
    assert dest.read_bytes() == b"vid"
    explicit = _deliver(master, tmp_path / "x" / "named.mp4", "ignored")
    assert explicit.name == "named.mp4" and explicit.exists()


def test_toddler_profile_loads_with_stricter_gate():
    prof = load_profile("toddler")
    assert prof.target_age == "2-6"
    assert prof.max_reading_grade == 2.5      # stricter than bedtime's 3.2
    assert prof.raw["tts_speed"] == 0.75
    assert prof.voices["narrator"] and "REPLACE" not in prof.voices["narrator"]
    assert prof.scary_words                    # safety list never empty


# ---- scene director chunking --------------------------------------------------

def test_scene_director_covers_all_lines_despite_inclusive_ends(
        fast_profile, tmp_path):
    """Live LLMs return inclusive line_end (python code sliced exclusively),
    which silently dropped the last line of every scene. Chunking must use
    consecutive line_starts and cover every line regardless of convention."""
    from kids_story_pipeline import nodes
    from kids_story_pipeline.state import Line, Scene

    lines = [Line(text=f"Line number {i} of the story.", role="narrator")
             for i in range(10)]
    state = PipelineState(run_id="sd", story_text="x", profile_name="bedtime")
    state.style_anchor = "style"
    state.scenes = [Scene(id=0, title="__all__", lines=lines)]

    class InclusiveEndLLM:
        def complete_json(self, system, prompt):
            return {"scenes": [  # inclusive ends + first start not at 0
                {"title": "a", "line_start": 1, "line_end": 3,
                 "image_prompt": "p1", "sfx_prompt": "s"},
                {"title": "b", "line_start": 4, "line_end": 6,
                 "image_prompt": "p2", "sfx_prompt": "s"},
                {"title": "c", "line_start": 7, "line_end": 9,
                 "image_prompt": "p3", "sfx_prompt": "s"},
            ]}

    class P:
        llm = InclusiveEndLLM()

    conf = nodes.scene_director(state, P(), fast_profile, tmp_path)
    assert conf == 1.0
    assert sum(len(sc.lines) for sc in state.scenes) == len(lines)
    texts = [l.text for sc in state.scenes for l in sc.lines]
    assert texts == [l.text for l in lines]          # order + no dupes
    assert all("style" in sc.image_prompt for sc in state.scenes)


def test_scene_director_rerun_after_pause_keeps_all_lines(fast_profile,
                                                          tmp_path):
    """Re-running scene_director (gate pause -> resume) must not shrink the
    script: the node overwrites state.scenes, so a second pass reads its own
    buckets. Flattening across all scenes keeps every line."""
    from kids_story_pipeline import nodes
    from kids_story_pipeline.state import Line, Scene

    lines = [Line(text=f"Short line {i}.", role="narrator") for i in range(8)]

    class SplitLLM:
        def complete_json(self, system, prompt):
            n = len(prompt.strip().splitlines())
            return {"scenes": [
                {"title": "a", "line_start": 0, "image_prompt": "p"},
                {"title": "b", "line_start": n // 2, "image_prompt": "p"},
            ]}

    class P:
        llm = SplitLLM()

    state = PipelineState(run_id="sd2", story_text="x", profile_name="bedtime")
    state.style_anchor = "style"
    state.scenes = [Scene(id=0, title="__all__", lines=lines)]
    assert nodes.scene_director(state, P(), fast_profile, tmp_path) == 1.0
    # simulate gate pause -> resume: node runs again on its own output
    assert nodes.scene_director(state, P(), fast_profile, tmp_path) == 1.0
    assert sum(len(sc.lines) for sc in state.scenes) == len(lines)


def test_animate_resume_skips_downloaded_hero_clips(fast_profile, tmp_path):
    """Hero renders are paid — a resume after a mid-node crash must not
    re-render clips whose raw file already landed on disk."""
    from PIL import Image
    from kids_story_pipeline import nodes
    from kids_story_pipeline.state import Line, Scene

    img = tmp_path / "scene.png"
    Image.new("RGB", (320, 180), (20, 30, 60)).save(img)
    state = PipelineState(run_id="an", story_text="x", profile_name="bedtime")
    sc = Scene(id=0, title="hero", lines=[Line(text="hi", role="narrator")],
               is_hero=True)
    sc.image_path = str(img)
    sc.audio_duration_s = 2.0
    state.scenes = [sc]

    calls = {"n": 0}

    class CountingVideo:
        def animate(self, image, motion_prompt, duration_s, out):
            calls["n"] += 1
            ff.make_kenburns_clip(image, duration_s, out,
                                  size=(320, 180), fps=12)
            return out

    class P:
        video = CountingVideo()

    nodes.animate(state, P(), fast_profile, tmp_path)
    assert calls["n"] == 1
    # crash-resume: raw exists, provider must not be called again
    nodes.animate(state, P(), fast_profile, tmp_path)
    assert calls["n"] == 1


def test_animate_hero_tail_lands_on_exact_duration(fast_profile, tmp_path):
    """When the hero render is shorter than the narration, the kenburns tail
    plus inner crossfade must land exactly on audio + scene crossfade —
    otherwise every hero scene shortens the video and drifts A/V sync."""
    from PIL import Image
    from kids_story_pipeline import nodes
    from kids_story_pipeline.state import Line, Scene

    img = tmp_path / "scene.png"
    Image.new("RGB", (640, 360), (20, 30, 60)).save(img)
    state = PipelineState(run_id="ht", story_text="x", profile_name="bedtime")
    sc = Scene(id=0, title="hero", lines=[Line(text="hi", role="narrator")],
               is_hero=True)
    sc.image_path = str(img)
    sc.audio_duration_s = 6.0
    state.scenes = [sc]

    class ShortVideo:  # renders only 2s of the needed 6.5s
        def animate(self, image, motion_prompt, duration_s, out):
            ff.make_kenburns_clip(image, 2.0, out, size=(640, 360), fps=12)
            return out

    class P:
        video = ShortVideo()

    nodes.animate(state, P(), fast_profile, tmp_path)
    expected = 6.0 + fast_profile.crossfade_s
    assert ff.probe_duration(state.scenes[0].clip_path) == pytest.approx(
        expected, abs=0.3)


def test_total_minutes_rounds_full_video_length(fast_profile):
    """101s narration + outro must label as 2 min, not floor to 1."""
    from kids_story_pipeline import nodes
    from kids_story_pipeline.state import Scene
    prof = load_profile("bedtime")          # 20s outro
    state = PipelineState(run_id="m", story_text="x", profile_name="bedtime")
    sc = Scene(id=0, title="s", lines=[])
    sc.audio_duration_s = 101.3
    state.scenes = [sc]
    assert nodes._total_minutes(state, prof) == 2
    sc.audio_duration_s = 5.0               # tiny video still says 1 min
    assert nodes._total_minutes(state, prof) == 1


# ---- vision QC ---------------------------------------------------------------

def _scene_with_image(tmp_path, sid=0, hero=False, size=(640, 360)):
    from PIL import Image as PILImage
    from kids_story_pipeline.state import Line, Scene
    img = tmp_path / "assets" / f"scene_{sid:02d}.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", size, (20, 30, 60)).save(img)
    sc = Scene(id=sid, title=f"s{sid}",
               lines=[Line(text="Toby bit the stick.", role="narrator")],
               is_hero=hero, image_prompt="old prompt, style")
    sc.image_path = str(img)
    sc.audio_duration_s = 2.0
    return sc


def test_vision_qc_rerolls_mismatched_image_and_invalidates_clip(
        fast_profile, tmp_path):
    """An image that shows the wrong action (holding vs biting) must be
    re-rolled with the reviewer's corrected prompt, and that scene's stale
    rendered clips deleted so animate re-renders them."""
    from kids_story_pipeline import nodes

    state = PipelineState(run_id="qc", story_text="x", profile_name="bedtime",
                          mock=False)
    state.style_anchor = "style"
    state.scenes = [_scene_with_image(tmp_path, 0, hero=True),
                    _scene_with_image(tmp_path, 1)]
    stale = tmp_path / "clips" / "scene_00_hero_raw.mp4"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")
    state.scenes[0].clip_path = str(stale)

    calls = {"reviews": 0, "gens": 0}

    class VisionLLM:
        def complete_json(self, system, prompt, images=None):
            assert "VISION_QC_TASK" in system and images
            assert "FULL STORY" in prompt        # continuity context present
            calls["reviews"] += 1
            # scene 0 fails once (wrong action), passes after re-roll
            if "scene_00" in str(images[0]) and calls["gens"] == 0:
                return {"matches": False,
                        "issues": ["turtle holds stick with flippers"],
                        "corrected_prompt": "turtle hanging by MOUTH from stick"}
            return {"matches": True, "issues": [], "corrected_prompt": ""}

    class CountingImages:
        def generate(self, prompt, out, reference=None, size=(640, 360)):
            calls["gens"] += 1
            from PIL import Image as PILImage
            PILImage.new("RGB", (640, 360), (99, 0, 0)).save(out)
            return out

    class P:
        llm = VisionLLM()
        images = CountingImages()

    conf = nodes.qc_visuals(state, P(), fast_profile, tmp_path)
    assert calls["gens"] == 1                       # only scene 0 re-rolled
    assert calls["reviews"] == 3                    # 0 fail, 0 pass, 1 pass
    assert "hanging by MOUTH" in state.scenes[0].image_prompt
    assert not stale.exists()                       # stale clip invalidated
    assert state.scenes[0].clip_path is None
    assert conf == 0.9                              # all pass -> capped at 0.9


def test_vision_qc_skipped_in_mock_mode(fast_profile, tmp_path):
    from kids_story_pipeline import nodes
    state = PipelineState(run_id="qcm", story_text="x",
                          profile_name="bedtime", mock=True)
    state.scenes = [_scene_with_image(tmp_path, 0)]

    class P:  # no llm/images at all — mock mode must not need them
        pass

    assert nodes.qc_visuals(state, P(), fast_profile, tmp_path) == 0.9


def test_image_gen_skips_existing_images(fast_profile, tmp_path):
    from kids_story_pipeline import nodes
    state = PipelineState(run_id="ig", story_text="x", profile_name="bedtime")
    state.scenes = [_scene_with_image(tmp_path, 0), _scene_with_image(tmp_path, 1)]
    (tmp_path / "assets" / "scene_01.png").unlink()   # scene 1 missing

    calls = {"gens": []}

    class CountingImages:
        def generate(self, prompt, out, reference=None, size=(640, 360)):
            calls["gens"].append(out.name)
            from PIL import Image as PILImage
            PILImage.new("RGB", (64, 36), (0, 0, 0)).save(out)
            return out

    class P:
        images = CountingImages()

    conf = nodes.image_gen(state, P(), fast_profile, tmp_path)
    assert calls["gens"] == ["scene_01.png"]          # existing not re-paid
    assert conf == 1.0


def test_animate_uses_scene_motion_prompt(fast_profile, tmp_path):
    from kids_story_pipeline import nodes
    state = PipelineState(run_id="mp", story_text="x", profile_name="bedtime")
    sc = _scene_with_image(tmp_path, 0, hero=True)
    sc.motion_prompt = "hangs by his mouth, flippers dangling, gentle sway"
    state.scenes = [sc]
    captured = {}

    class CapturingVideo:
        def animate(self, image, motion_prompt, duration_s, out):
            captured["motion"] = motion_prompt
            ff.make_kenburns_clip(image, duration_s, out,
                                  size=(320, 180), fps=12)
            return out

    class P:
        video = CapturingVideo()

    nodes.animate(state, P(), fast_profile, tmp_path)
    assert captured["motion"] == sc.motion_prompt


# ---- gate pause / resume ----------------------------------------------------

def test_gate_pauses_and_resume_approves(fast_profile, tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "RUNS_DIR", tmp_path)
    scary = SAMPLE + "\n\nThen a monster with a knife made everyone scream."
    with pytest.raises(graph.GatePaused) as exc:
        graph.start_run(scary, fast_profile, mock=True, run_id="gated-run")
    assert exc.value.node == "audience_gate"
    saved = PipelineState.load(tmp_path / "gated-run")
    assert saved.status == "paused"
    assert (tmp_path / "gated-run" / "PENDING_APPROVAL.txt").exists()

    # human approves -> run continues to completion
    state = graph.resume_run("gated-run", fast_profile, approve=True)
    assert state.status == "done"
    assert Path(state.master_video_path).exists()


def test_no_gate_flag_runs_hands_off(tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "RUNS_DIR", tmp_path)
    prof = load_profile("bedtime", overrides={**FAST, "gate_enabled": False})
    scary = SAMPLE + "\n\nThen a monster appeared."
    state = graph.start_run(scary, prof, mock=True, run_id="nogate-run")
    assert state.status == "done"


# ---- end-to-end mock ---------------------------------------------------------

def test_e2e_mock_produces_all_deliverables(fast_profile, tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "RUNS_DIR", tmp_path)
    state = graph.start_run(SAMPLE, fast_profile, mock=True, run_id="e2e-run")
    assert state.status == "done"

    master = Path(state.master_video_path)
    assert master.exists() and sorted(ff.probe_streams(master)) == ["audio", "video"]

    # scene clips are padded by one crossfade each, so the xfade overlaps
    # consume the padding and video length == narration + outro (A/V sync:
    # every scene's visuals start exactly when its narration starts)
    narration = sum(sc.audio_duration_s for sc in state.scenes)
    expected = narration + fast_profile.outro_s
    assert ff.probe_duration(master) == pytest.approx(expected, abs=1.5)

    assert Path(state.shorts_path).exists()
    assert ff.probe_duration(state.shorts_path) <= fast_profile.shorts_max_s + 0.5
    assert Path(state.thumbnail_path).exists()

    meta = json.loads(Path(state.metadata_path).read_text())
    assert meta["made_for_kids"] is True
    assert "Bedtime Story" in meta["title"]

    # hero scene actually used the video provider path
    assert any(sc.is_hero for sc in state.scenes)

    # checkpoint recorded every node
    assert len(state.completed_nodes) == 14


def test_state_roundtrip(tmp_path):
    st = PipelineState(run_id="x", story_text="hello world", profile_name="bedtime")
    st.save(tmp_path)
    back = PipelineState.load(tmp_path)
    assert back.run_id == "x" and back.story_text == "hello world"
