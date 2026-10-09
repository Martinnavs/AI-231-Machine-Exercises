"""Music service: a simulated player (state only -- no audio in this pass).

Five named tracks; a `STOPPED`/`PLAYING`/`PAUSED` state machine; a 0-100
volume (default 50) stepped in tens and capped, in the lights style.
`STOP` halts **and** resets to the first track -- a distinct state from
`PAUSED` (feature `ui-site`, decision Q9). `NEXT` wraps around the track
list and is valid while playing or paused; it is ignored while stopped
(there is no playing track to advance from). Volume is an orthogonal
attribute (like the lights' color): settable in any state, including
stopped.

Audio output is a seam: with no `player` (the default, used by tests)
nothing plays and the panel is state-only; `services.music_player`
supplies an ffplay-backed player for the real app.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable

# The simulated playlist (decision Q9: five named tracks).
TRACKS = ("Midnight Drive", "Ocean Calm", "City Lights", "Morning Fog", "Starfall")


class MusicState(Enum):
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


class MusicEvent(Enum):
    PLAY = auto()
    PAUSE = auto()
    NEXT = auto()
    STOP = auto()


class MusicService:
    """Explicit `(state, event) -> next_state` table + ignore-on-unhandled."""

    # Soft pause (wake word heard): resume this long after the listening
    # period ends so a PAUSE/STOP that follows can cancel it; the cap
    # guards against a `passive` that never arrives.
    RESUME_GRACE_S = 0.8
    SOFT_PAUSE_MAX_S = 10.0

    DEFAULT_VOLUME = 50
    VOLUME_STEP = 10
    VOLUME_MIN = 0
    VOLUME_MAX = 100

    TRANSITIONS = {
        # Play starts (from stop) or resumes (from pause).
        (MusicState.STOPPED, MusicEvent.PLAY): MusicState.PLAYING,
        (MusicState.PAUSED, MusicEvent.PLAY): MusicState.PLAYING,
        # Pause only makes sense while playing.
        (MusicState.PLAYING, MusicEvent.PAUSE): MusicState.PAUSED,
        # Next track (wraps around), in either an active state.
        (MusicState.PLAYING, MusicEvent.NEXT): MusicState.PLAYING,
        (MusicState.PAUSED, MusicEvent.NEXT): MusicState.PAUSED,
        # Stop from any state: halts + resets to the first track.
        (MusicState.STOPPED, MusicEvent.STOP): MusicState.STOPPED,
        (MusicState.PLAYING, MusicEvent.STOP): MusicState.STOPPED,
        (MusicState.PAUSED, MusicEvent.STOP): MusicState.STOPPED,
        # Ignored (no entry): PLAY while PLAYING, PAUSE while PAUSED/STOPPED,
        # NEXT while STOPPED.
    }

    def __init__(self, tracks: tuple[str, ...] = TRACKS, player=None, volume: int | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """`player` is the audio seam (`services.music_player`); None keeps
        the service state-only."""
        self.tracks = tracks
        self.player = player
        self.state = MusicState.STOPPED
        self.track_index = 0
        self.volume = self.DEFAULT_VOLUME if volume is None else volume
        self._clock = clock
        self.soft_paused = False
        self._resume_at = 0.0

    @property
    def track(self) -> str:
        return self.tracks[self.track_index]

    def _drive_player(self, event: MusicEvent, prev: MusicState) -> None:
        if self.player is None:
            return
        try:
            if event is MusicEvent.PLAY:
                if prev is MusicState.PAUSED:
                    self.player.resume()
                else:
                    self.player.start(self.track_index, self.volume)
            elif event is MusicEvent.PAUSE:
                self.player.pause()
            elif event is MusicEvent.NEXT:
                self.player.start(self.track_index, self.volume, paused=self.state is MusicState.PAUSED)
            elif event is MusicEvent.STOP:
                self.player.stop()
        except OSError:
            pass  # no audio device/binary: keep the UI state, drop the sound

    def on_listening(self, listening: str) -> None:
        """Wake word heard -> soft-pause (audio only; the state stays PLAYING).
        Listening over -> schedule the resume. A music transition that
        lands in between (PAUSE, STOP, NEXT) clears the soft pause."""
        if listening == "active":
            if self.state is MusicState.PLAYING and self.player is not None and not self.soft_paused:
                try:
                    self.player.pause()
                except OSError:
                    return
                self.soft_paused = True
            if self.soft_paused:
                self._resume_at = self._clock() + self.SOFT_PAUSE_MAX_S
        elif self.soft_paused:
            self._resume_at = self._clock() + self.RESUME_GRACE_S

    def tick(self) -> None:
        """Resume a due soft pause; advance when the track ends on its own."""
        if self.soft_paused and self._clock() >= self._resume_at:
            self.soft_paused = False
            try:
                self.player.resume()
            except OSError:
                pass
            return
        if self.player is not None and self.state is MusicState.PLAYING and self.player.finished():
            self.next_track()

    def dispatch(self, event: MusicEvent) -> tuple[bool, str]:
        """Apply `event` if a transition is defined; else ignore."""
        next_state = self.TRANSITIONS.get((self.state, event))
        if next_state is None:
            return False, (
                f"Ignored: cannot {event.name.lower()} while {self.state.name.lower()}"
            )
        prev = self.state
        self.soft_paused = False  # an applied command supersedes the soft pause
        self.state = next_state
        if event is MusicEvent.NEXT:
            self.track_index = (self.track_index + 1) % len(self.tracks)
        elif event is MusicEvent.STOP:
            self.track_index = 0
        self._drive_player(event, prev)
        return True, (
            f"[{event.name}] -> {self.state.name} "
            f"(track {self.track_index + 1}: {self.track})"
        )

    def play(self) -> tuple[bool, str]:
        return self.dispatch(MusicEvent.PLAY)

    def pause(self) -> tuple[bool, str]:
        return self.dispatch(MusicEvent.PAUSE)

    def next_track(self) -> tuple[bool, str]:
        return self.dispatch(MusicEvent.NEXT)

    def stop(self) -> tuple[bool, str]:
        return self.dispatch(MusicEvent.STOP)

    def _change_volume(self, delta: int) -> tuple[bool, str]:
        new_volume = self.volume + delta
        if new_volume < self.VOLUME_MIN or new_volume > self.VOLUME_MAX:
            return False, f"Ignored: volume already at {self.volume} (limit)"
        self.volume = new_volume
        if self.player is not None:
            try:
                self.player.set_volume(new_volume)
            except OSError:
                pass
        return True, f"volume -> {new_volume}"

    def volume_up(self) -> tuple[bool, str]:
        return self._change_volume(+self.VOLUME_STEP)

    def volume_down(self) -> tuple[bool, str]:
        return self._change_volume(-self.VOLUME_STEP)

    def to_dict(self) -> dict:
        return {
            "state": self.state.name,
            "track_index": self.track_index,
            "track": self.track,
            "track_total": len(self.tracks),
            "volume": self.volume,
            "soft_paused": self.soft_paused,
        }
