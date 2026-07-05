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

    # video length tracks narration + outro (crossfades subtract a little)
    narration = sum(sc.audio_duration_s for sc in state.scenes)
    n_fades = len(state.scenes)  # scenes + outro joined with n fades
    expected = narration + fast_profile.outro_s - n_fades * fast_profile.crossfade_s
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
