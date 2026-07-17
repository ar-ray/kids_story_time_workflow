"""ffmpeg wrappers. All subprocess calls use arg lists (no shell)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class FfmpegError(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FfmpegError(f"{' '.join(cmd[:6])}... failed:\n{proc.stderr[-2000:]}")


def probe_duration(path: Path | str) -> float:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(proc.stdout)["format"]["duration"])


def probe_streams(path: Path | str) -> list[str]:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return [s["codec_type"] for s in json.loads(proc.stdout)["streams"]]


# ---- video -----------------------------------------------------------------

def make_kenburns_clip(image: Path, duration_s: float, out: Path,
                       size: tuple[int, int] = (1280, 720), fps: int = 25,
                       zoom_rate: float = 0.0004, max_zoom: float = 1.06) -> Path:
    """Silent H.264 clip from a still image with a slow center zoom.

    zoompan quantizes its crop window to whole source pixels, which reads as
    shake on slow zooms from a ~1080p source — supersample the input to 4x
    the output width so each step is sub-output-pixel and the motion smooth.
    """
    frames = max(1, int(round(duration_s * fps)))
    w, h = size
    vf = (
        f"scale={w * 4}:-2,"
        f"zoompan=z='min(zoom+{zoom_rate},{max_zoom})':d={frames}"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps},"
        "format=yuv420p"
    )
    _run([FFMPEG, "-y", "-i", str(image), "-vf", vf, "-frames:v", str(frames),
          "-c:v", "libx264", "-preset", "veryfast", "-an", str(out)])
    return out


def normalize_clip(src: Path, out: Path, size: tuple[int, int] = (1280, 720),
                   fps: int = 25) -> Path:
    """Re-encode any clip to the uniform format assembly expects."""
    w, h = size
    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p"
    _run([FFMPEG, "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264",
          "-preset", "veryfast", "-an", str(out)])
    return out


def xfade_concat(clips: list[Path], out: Path, fade_s: float = 1.2) -> Path:
    """Join uniform clips with crossfades. offset_i = sum(d_1..i) - i*fade."""
    if not clips:
        raise ValueError("no clips to concat")
    if len(clips) == 1:
        shutil.copy(clips[0], out)
        return out
    durations = [probe_duration(c) for c in clips]
    cmd: list[str] = [FFMPEG, "-y"]
    for c in clips:
        cmd += ["-i", str(c)]
    parts: list[str] = []
    prev = "[0:v]"
    offset = 0.0
    for i in range(1, len(clips)):
        offset += durations[i - 1] - fade_s
        label = f"[v{i}]"
        parts.append(
            f"{prev}[{i}:v]xfade=transition=fade:duration={fade_s}:offset={offset:.3f}{label}"
        )
        prev = label
    cmd += ["-filter_complex", ";".join(parts), "-map", prev,
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)]
    _run(cmd)
    return out


def extract_frame(video: Path, at_s: float, out: Path) -> Path:
    _run([FFMPEG, "-y", "-ss", f"{max(0.0, at_s):.3f}", "-i", str(video),
          "-frames:v", "1", str(out)])
    return out


def mux(video: Path, audio: Path, out: Path) -> Path:
    _run([FFMPEG, "-y", "-i", str(video), "-i", str(audio),
          "-map", "0:v", "-map", "1:a", "-c:v", "copy",
          "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)])
    return out


def shorts_cutdown(master: Path, out: Path, max_s: float = 60.0) -> Path:
    """Center-crop 9:16 vertical teaser from the master."""
    dur = min(max_s, probe_duration(master))
    vf = "crop=ih*9/16:ih,scale=1080:1920,format=yuv420p"
    _run([FFMPEG, "-y", "-i", str(master), "-t", f"{dur:.3f}", "-vf", vf,
          "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out)])
    return out


# ---- audio -----------------------------------------------------------------

def synth_tone(out: Path, duration_s: float, freq: int = 220, sample_rate: int = 24000) -> Path:
    """Sine placeholder audio (mock TTS / mock music)."""
    _run([FFMPEG, "-y", "-f", "lavfi",
          "-i", f"sine=frequency={freq}:duration={duration_s:.3f}",
          "-ar", str(sample_rate), "-ac", "1", str(out)])
    return out


def synth_silence(out: Path, duration_s: float, sample_rate: int = 24000) -> Path:
    _run([FFMPEG, "-y", "-f", "lavfi",
          "-i", f"anullsrc=r={sample_rate}:cl=mono",
          "-t", f"{duration_s:.3f}", str(out)])
    return out


def pad_audio(src: Path, out: Path, pad_s: float) -> Path:
    _run([FFMPEG, "-y", "-i", str(src), "-af", f"apad=pad_dur={pad_s:.3f}", str(out)])
    return out


def concat_audio(files: list[Path], out: Path) -> Path:
    cmd: list[str] = [FFMPEG, "-y"]
    for f in files:
        cmd += ["-i", str(f)]
    labels = "".join(f"[{i}:a]" for i in range(len(files)))
    fc = f"{labels}concat=n={len(files)}:v=0:a=1[a]"
    cmd += ["-filter_complex", fc, "-map", "[a]", str(out)]
    _run(cmd)
    return out


def mix_music(narration: Path, music: Path, out: Path, music_gain: float = 0.10) -> Path:
    fc = (f"[1:a]volume={music_gain}[m];"
          f"[0:a][m]amix=inputs=2:duration=longest:normalize=0,"
          f"loudnorm=I=-18:TP=-2[a]")
    _run([FFMPEG, "-y", "-i", str(narration), "-i", str(music),
          "-filter_complex", fc, "-map", "[a]", str(out)])
    return out
