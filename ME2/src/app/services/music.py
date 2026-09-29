"""Music service: a simulated player (state only -- no audio in this pass).

Five named tracks; a `STOPPED`/`PLAYING`/`PAUSED` state machine; a 0-100
volume (default 50) stepped in tens and capped, in the lights style.
`STOP` halts **and** resets to the first track -- a distinct state from
`PAUSED` (feature `ui-site`, decision Q9). `NEXT` wraps around the track
list and is valid while playing or paused; it is ignored while stopped
(there is no playing track to advance from). Volume is an orthogonal
attribute (like the lights' color): settable in any state, including
stopped.

Audio output is a deliberate seam: nothing here plays a sound, so the
panel is state-only for now (decision: "audio later -- leave seam").
"""

from __future__ import annotations

from enum import Enum, auto

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

    def __init__(self) -> None:
        self.state = MusicState.STOPPED
        self.track_index = 0
        self.volume = self.DEFAULT_VOLUME

    @property
    def track(self) -> str:
        return TRACKS[self.track_index]

    def dispatch(self, event: MusicEvent) -> tuple[bool, str]:
        """Apply `event` if a transition is defined; else ignore."""
        next_state = self.TRANSITIONS.get((self.state, event))
        if next_state is None:
            return False, (
                f"Ignored: cannot {event.name.lower()} while {self.state.name.lower()}"
            )
        self.state = next_state
        if event is MusicEvent.NEXT:
            self.track_index = (self.track_index + 1) % len(TRACKS)
        elif event is MusicEvent.STOP:
            self.track_index = 0
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
            "track_total": len(TRACKS),
            "volume": self.volume,
        }
