"""Music service: STOPPED/PLAYING/PAUSED machine, wrapping next,
stop-resets-to-track-one, capped volume (decision Q9)."""

from app.services.music import TRACKS, MusicEvent, MusicService, MusicState


def test_defaults_stopped_track_one_volume_50():
    svc = MusicService()
    assert svc.state is MusicState.STOPPED
    assert svc.track_index == 0
    assert svc.volume == 50


def test_five_named_tracks():
    assert len(TRACKS) == 5
    assert len(set(TRACKS)) == 5


def test_play_from_stopped_starts_at_track_one():
    svc = MusicService()
    changed, _msg = svc.play()
    assert changed
    assert svc.state is MusicState.PLAYING
    assert svc.track_index == 0


def test_pause_and_resume():
    svc = MusicService()
    svc.play()
    assert svc.pause()[0]
    assert svc.state is MusicState.PAUSED
    svc.track_index = 2
    assert svc.play()[0]  # resume, keep the track
    assert svc.state is MusicState.PLAYING
    assert svc.track_index == 2


def test_play_while_playing_ignored():
    svc = MusicService()
    svc.play()
    changed, msg = svc.play()
    assert not changed
    assert msg.startswith("Ignored")


def test_pause_while_paused_or_stopped_ignored():
    svc = MusicService()
    changed, msg = svc.pause()
    assert not changed and msg.startswith("Ignored")
    svc.play()
    svc.pause()
    changed, msg = svc.pause()
    assert not changed and msg.startswith("Ignored")


def test_next_wraps_around():
    svc = MusicService()
    svc.play()
    for expected in (1, 2, 3, 4):
        assert svc.next_track()[0]
        assert svc.track_index == expected
    assert svc.next_track()[0]  # wrap
    assert svc.track_index == 0
    assert svc.state is MusicState.PLAYING


def test_next_from_paused_advances_and_stays_paused():
    svc = MusicService()
    svc.play()
    svc.pause()
    assert svc.next_track()[0]
    assert svc.track_index == 1
    assert svc.state is MusicState.PAUSED


def test_next_while_stopped_ignored():
    svc = MusicService()
    changed, msg = svc.next_track()
    assert not changed
    assert msg.startswith("Ignored")


def test_stop_from_each_state_resets_to_track_one():
    for st in (MusicState.PLAYING, MusicState.PAUSED, MusicState.STOPPED):
        svc = MusicService()
        svc.state = st
        svc.track_index = 3
        changed, _msg = svc.stop()
        assert changed
        assert svc.state is MusicState.STOPPED
        assert svc.track_index == 0


def test_volume_steps_in_tens_and_caps():
    svc = MusicService()
    for expected in (60, 70, 80, 90, 100):
        assert svc.volume_up()[0]
        assert svc.volume == expected
    changed, msg = svc.volume_up()  # capped at 100
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.volume == 100
    for expected in (90, 80, 70, 60, 50, 40, 30, 20, 10, 0):
        assert svc.volume_down()[0]
        assert svc.volume == expected
    changed, msg = svc.volume_down()  # floored at 0
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.volume == 0


def test_volume_works_while_stopped():
    svc = MusicService()
    assert svc.volume_up()[0]
    assert svc.volume == 60
    assert svc.state is MusicState.STOPPED


def test_every_unmapped_state_event_pair_is_ignored():
    for st in MusicState:
        for ev in (MusicEvent.PLAY, MusicEvent.PAUSE, MusicEvent.NEXT, MusicEvent.STOP):
            svc = MusicService()
            svc.state = st
            if (st, ev) in MusicService.TRANSITIONS:
                continue
            changed, msg = svc.dispatch(ev)
            assert not changed, (st, ev)
            assert msg.startswith("Ignored")
            assert svc.state is st
