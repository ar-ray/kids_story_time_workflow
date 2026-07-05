"""Checkpointed linear graph runner with an adaptive approval gate.

Note on orchestration choice: v1's graph is strictly linear, so this is a
dependency-free runner with LangGraph-compatible semantics (per-node
checkpointing, resumable interrupts). If/when the graph gains branching
(e.g. QC re-roll loops calling back into image_gen), swap this file for a
LangGraph StateGraph — the node contract is already compatible.
"""
from __future__ import annotations

from pathlib import Path

from .config import Profile, PROJECT_ROOT
from .nodes import NODES
from .providers import build_providers
from .state import PipelineState, new_run_id

RUNS_DIR = PROJECT_ROOT / "runs"


class GatePaused(Exception):
    def __init__(self, node: str, confidence: float, run_id: str):
        self.node, self.confidence, self.run_id = node, confidence, run_id
        super().__init__(
            f"Gate paused at '{node}' (confidence {confidence}). "
            f"Review runs/{run_id}/ then resume with: "
            f"python -m kids_story_pipeline resume {run_id} --approve")


def run_dir_for(run_id: str) -> Path:
    return RUNS_DIR / run_id


def start_run(story_text: str, profile: Profile, mock: bool = True,
              run_id: str | None = None) -> PipelineState:
    state = PipelineState(run_id=run_id or new_run_id(), story_text=story_text,
                          profile_name=profile.name, mock=mock)
    state.save(run_dir_for(state.run_id))
    return execute(state, profile)


def resume_run(run_id: str, profile: Profile, approve: bool = False) -> PipelineState:
    state = PipelineState.load(run_dir_for(run_id))
    if state.pending_gate and approve:
        state.approved_gates.append(state.pending_gate["node"])
        state.pending_gate = None
        state.status = "running"
    return execute(state, profile)


def execute(state: PipelineState, profile: Profile) -> PipelineState:
    if state.pending_gate:
        raise GatePaused(state.pending_gate["node"],
                         state.pending_gate["confidence"], state.run_id)
    providers = build_providers(state.mock, profile)
    rd = run_dir_for(state.run_id)
    state.status = "running"  # a resumed 'failed' run is running again
    try:
        for name, fn, gated in NODES:
            if name in state.completed_nodes:
                continue
            confidence = float(fn(state, providers, profile, rd))
            state.confidences[name] = confidence
            low = confidence < profile.gate_threshold
            if gated and profile.gate_enabled and low and name not in state.approved_gates:
                state.status = "paused"
                state.pending_gate = {"node": name, "confidence": confidence}
                state.save(rd)
                (rd / "PENDING_APPROVAL.txt").write_text(
                    f"node={name}\nconfidence={confidence}\n"
                    "Inspect state.json / assets, then resume with --approve.\n")
                raise GatePaused(name, confidence, state.run_id)
            if low:
                state.notes.append(f"{name}: low confidence {confidence} (not gated)")
            state.completed_nodes.append(name)
            state.save(rd)
        state.status = "done"
        state.save(rd)
        return state
    except GatePaused:
        raise
    except Exception:
        state.status = "failed"
        state.save(rd)
        raise
