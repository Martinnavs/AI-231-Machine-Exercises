"""MusicService <-> player seam, with a fake player (no audio)."""

from pathlib import Path

from app.services.music import MusicService
from app.services.music_player import FfplayPlayer, default_music, discover_tracks, track_title


class FakePlayer:
    def __init__(self):
        self.calls = []
        self.done = False

    def start(self, index, volume, paused=False):
        self.calls.append(("start", index, volume, paused))

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def stop(self):
        self.calls.append(("stop",))

    def set_volume(self, volume):
        self.calls.append(("volume", volume))

    def finished(self):
        return self.done


def make():
    p = FakePlayer()
    return MusicService(tracks=("a", "b", "c"), player=p, volume=30), p


def test_play_starts_at_configured_volume_and_resume_after_pause():
    svc, p = make()
    svc.play()
    svc.pause()
    svc.play()
    assert p.calls == [("start", 0, 30, False), ("pause",), ("resume",)]


def test_next_restarts_track_and_stays_paused_when_paused():
    svc, p = make()
    svc.play()
    svc.next_track()
    svc.pause()
    svc.next_track()
    assert p.calls[1] == ("start", 1, 30, False)
    assert p.calls[-1] == ("start", 2, 30, True)


def test_stop_and_volume_reach_player_and_ignored_events_do_not():
    svc, p = make()
    svc.pause()  # ignored while stopped
    svc.volume_up()
    svc.play()
    svc.stop()
    assert p.calls == [("volume", 40), ("start", 0, 40, False), ("stop",)]


def test_track_end_advances_only_while_playing():
    svc, p = make()
    p.done = True
    svc.tick()
    assert svc.track_index == 0  # stopped: nothing happens
    svc.play()
    svc.tick()
    assert svc.track_index == 1 and p.calls[-1] == ("start", 1, 30, False)


def test_default_service_has_no_player():
    svc = MusicService()
    assert svc.player is None and svc.volume == 50 and len(svc.tracks) == 5
    svc.play()
    svc.tick()


def test_discover_and_titles(tmp_path):
    (tmp_path / "b.mp3").write_bytes(b"")
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    found = discover_tracks(tmp_path)
    assert [track_title(p) for p in found] == ["a", "b"]
    assert discover_tracks(tmp_path / "missing") == []


def test_default_music_uses_repo_tracks_at_30(monkeypatch):
    monkeypatch.setattr("app.services.music_player.shutil.which", lambda _n: "/usr/bin/ffplay")
    monkeypatch.setattr("app.services.music_player.discover_tracks", lambda: [Path("x/One.mp3")])
    svc = default_music()
    assert svc.tracks == ("One",) and svc.volume == 30 and isinstance(svc.player, FfplayPlayer)


# --- soft pause on wake word ------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_soft():
    p, c = FakePlayer(), Clock()
    svc = MusicService(tracks=("a", "b", "c"), player=p, volume=30, clock=c)
    svc.play()
    p.calls.clear()
    return svc, p, c


def test_wake_word_soft_pauses_and_resumes_after_grace():
    svc, p, c = make_soft()
    svc.on_listening("active")
    assert p.calls == [("pause",)]
    assert svc.state.name == "PLAYING" and svc.to_dict()["soft_paused"] is True
    svc.on_listening("passive")
    svc.tick()
    assert p.calls == [("pause",)]  # still inside the grace
    c.t += svc.RESUME_GRACE_S + 0.01
    svc.tick()
    assert p.calls == [("pause",), ("resume",)] and not svc.soft_paused


def test_pause_or_stop_after_listening_cancels_the_resume():
    for command, expected in (("pause", "PAUSED"), ("stop", "STOPPED")):
        svc, p, c = make_soft()
        svc.on_listening("active")
        svc.on_listening("passive")
        getattr(svc, command)()
        c.t += 5
        svc.tick()
        assert ("resume",) not in p.calls
        assert svc.state.name == expected and not svc.soft_paused


def test_ignored_command_still_resumes_and_next_plays_new_track():
    svc, p, c = make_soft()
    svc.on_listening("active")
    svc.on_listening("passive")
    assert svc.play()[0] is False  # PLAY while PLAYING is ignored
    c.t += 1
    svc.tick()
    assert p.calls[-1] == ("resume",)
    svc.on_listening("active")
    svc.on_listening("passive")
    svc.next_track()
    assert p.calls[-1] == ("start", 1, 30, False) and not svc.soft_paused


def test_missing_passive_resumes_after_the_cap_and_paused_music_is_untouched():
    svc, p, c = make_soft()
    svc.on_listening("active")
    c.t += svc.SOFT_PAUSE_MAX_S + 1
    svc.tick()
    assert p.calls == [("pause",), ("resume",)]
    svc.pause()
    p.calls.clear()
    svc.on_listening("active")  # already paused by the user: leave it alone
    svc.on_listening("passive")
    c.t += 5
    svc.tick()
    assert p.calls == [] and svc.state.name == "PAUSED"
