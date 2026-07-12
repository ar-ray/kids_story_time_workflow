"""Pipeline state: JSON-serializable dataclasses passed between nodes."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class Line:
    """One narration line. role: 'narrator' or a character voice key."""
    text: str
    role: str = "narrator"


@dataclass
class Scene:
    id: int
    title: str
    lines: list[Line] = field(default_factory=list)
    image_prompt: str = ""
    sfx_prompt: str = ""
    motion_prompt: str = ""         # action-faithful animation direction
    is_hero: bool = False           # hero scenes get image-to-video animation
    audio_path: Optional[str] = None
    audio_duration_s: float = 0.0   # padded narration duration for this scene
    image_path: Optional[str] = None
    clip_path: Optional[str] = None  # normalized video clip used in assembly

    @property
    def narration_text(self) -> str:
        return " ".join(l.text for l in self.lines)


@dataclass
class PipelineState:
    run_id: str
    story_text: str
    profile_name: str
    mock: bool = True
    title: str = ""
    refrain: str = ""
    style_anchor: str = ""
    character_anchor: str = ""
    scenes: list[Scene] = field(default_factory=list)
    character_ref_path: Optional[str] = None
    music_path: Optional[str] = None
    master_video_path: Optional[str] = None
    shorts_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    metadata_path: Optional[str] = None
    confidences: dict[str, float] = field(default_factory=dict)
    completed_nodes: list[str] = field(default_factory=list)
    status: str = "running"          # running | paused | done | failed
    pending_gate: Optional[dict[str, Any]] = None
    approved_gates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ---- persistence -------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "PipelineState":
        d = json.loads(raw)
        scenes = []
        for s in d.get("scenes", []):
            lines = [Line(**l) for l in s.pop("lines", [])]
            scenes.append(Scene(lines=lines, **s))
        d["scenes"] = scenes
        return cls(**d)

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "state.json").write_text(self.to_json())

    @classmethod
    def load(cls, run_dir: Path) -> "PipelineState":
        return cls.from_json((run_dir / "state.json").read_text())


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
