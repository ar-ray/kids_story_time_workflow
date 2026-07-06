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
