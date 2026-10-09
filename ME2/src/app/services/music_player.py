"""Real audio for the music service: one `ffplay` subprocess per track.

`MusicService` talks to a small player seam (`start`, `pause`, `resume`,
`stop`, `set_volume`, `finished`); the default is no player (state-only,
which is what the tests use). `FfplayPlayer` plays through the system
default output (PipeWire here; set `ME2_MUSIC_AUDIODEV` for a specific
ALSA device). Pause is SIGSTOP/SIGCONT; a volume change restarts ffplay
at the current position, since ffplay's volume is fixed at launch.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

MUSIC_DIR = Path(__file__).resolve().parents[3] / "extras" / "music"
AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
DEFAULT_PLAY_VOLUME = 30  # percent, the demo's start level


def discover_tracks(directory: Path = MUSIC_DIR) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)


def track_title(path: Path) -> str:
    return path.stem


def _die_with_parent() -> None:
    """Linux: SIGKILL ffplay if the UI server dies, so no orphan keeps playing
    (SIGKILL because a SIGSTOPped process would not act on SIGTERM)."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except OSError:
        pass


class FfplayPlayer:
    def __init__(self, paths: list[Path], ffplay: str | None = None) -> None:
        self._paths = paths
        self._ffplay = ffplay or shutil.which("ffplay") or "ffplay"
        self._proc: subprocess.Popen | None = None
        self._index = 0
        self._volume = DEFAULT_PLAY_VOLUME
        self._paused = False
        self._position = 0.0  # seconds played before the current run
        self._run_started = 0.0

    # -- internals -----------------------------------------------------
    def _elapsed(self) -> float:
        if self._proc is None or self._paused:
            return self._position
        return self._position + (time.monotonic() - self._run_started)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGCONT)  # a stopped process ignores SIGTERM
            proc.terminate()
            proc.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            proc.kill()

    def _spawn(self, seek: float = 0.0) -> None:
        cmd = [self._ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
               "-volume", str(self._volume)]
        if seek > 0:
            cmd += ["-ss", f"{seek:.2f}"]
        cmd.append(str(self._paths[self._index]))
        env = dict(os.environ)
        if os.environ.get("ME2_MUSIC_AUDIODEV"):
            env.update(SDL_AUDIODRIVER="alsa", AUDIODEV=os.environ["ME2_MUSIC_AUDIODEV"])
        self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, env=env,
                                      preexec_fn=_die_with_parent)
        self._position = seek
        self._run_started = time.monotonic()

    # -- player seam ---------------------------------------------------
    def start(self, index: int, volume: int, paused: bool = False) -> None:
        self._kill()
        self._index, self._volume, self._paused = index, volume, False
        self._spawn()
        if paused:
            self.pause()

    def pause(self) -> None:
        if self._proc is not None and not self._paused:
            self._position = self._elapsed()
            self._paused = True
            self._proc.send_signal(signal.SIGSTOP)

    def resume(self) -> None:
        if self._proc is not None and self._paused:
            self._paused = False
            self._run_started = time.monotonic()
            self._proc.send_signal(signal.SIGCONT)

    def stop(self) -> None:
        self._kill()
        self._paused = False
        self._position = 0.0

    def set_volume(self, volume: int) -> None:
        self._volume = volume
        if self._proc is None:
            return
        pos, was_paused = self._elapsed(), self._paused
        self._kill()
        self._paused = False
        self._spawn(seek=pos)
        if was_paused:
            self.pause()

    def finished(self) -> bool:
        """True once the current track ended on its own (not paused/stopped)."""
        return self._proc is not None and not self._paused and self._proc.poll() is not None


def default_music():
    """The real-audio `MusicService` when `extras/music` has tracks and
    ffplay exists; otherwise the simulated one (no sound)."""
    from .music import MusicService

    paths = discover_tracks()
    if not paths or shutil.which("ffplay") is None:
        return MusicService()
    return MusicService(
        tracks=tuple(track_title(p) for p in paths),
        player=FfplayPlayer(paths),
        volume=DEFAULT_PLAY_VOLUME,
    )
