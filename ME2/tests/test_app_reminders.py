"""Reminders service: grammar-restricted, append-only, log-and-list with
a collapsed/expanded panel (decision Q11)."""

from datetime import datetime, timedelta

from app.services.reminders import ReminderService


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: int) -> None:
        self.t += timedelta(seconds=seconds)


def test_defaults_empty_and_collapsed():
    svc = ReminderService(now=lambda: datetime(2026, 9, 29, 8, 0, 0))
    data = svc.to_dict()
    assert data["expanded"] is False
    assert data["items"] == []
    assert data["latest"] is None


def test_add_stamps_the_machine_time_when_logged():
    clock = Clock(datetime(2026, 9, 29, 8, 0, 0))
    svc = ReminderService(now=clock)
    changed, _msg = svc.add("study")
    assert changed
    assert svc.latest["task"] == "study"
    assert svc.latest["logged_at"] == "08:00:00"


def test_add_rejects_tasks_outside_the_grammar_choices():
    svc = ReminderService(now=lambda: datetime(2026, 9, 29, 8, 0, 0))
    changed, msg = svc.add("walk the dog")
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.items == []


def test_items_are_append_only_and_listed_newest_first():
    clock = Clock(datetime(2026, 9, 29, 8, 0, 0))
    svc = ReminderService(now=clock)
    svc.add("study")
    clock.advance(5)
    svc.add("exercise")
    assert [item["task"] for item in svc.items] == ["exercise", "study"]
    assert [item["logged_at"] for item in svc.items] == ["08:00:05", "08:00:00"]
    assert svc.latest["task"] == "exercise"
    assert svc.items[0]["id"] == 2  # stable ids, chronological order


def test_expand_and_collapse():
    svc = ReminderService(now=lambda: datetime(2026, 9, 29, 8, 0, 0))
    assert svc.expand()[0]
    assert svc.expanded is True
    assert svc.collapse()[0]
    assert svc.expanded is False


def test_latest_tracks_the_newest_after_more_adds():
    clock = Clock(datetime(2026, 9, 29, 8, 0, 0))
    svc = ReminderService(now=clock)
    svc.add("drink water")
    assert svc.latest["task"] == "drink water"
    clock.advance(1)
    svc.add("study")
    assert svc.latest["task"] == "study"
