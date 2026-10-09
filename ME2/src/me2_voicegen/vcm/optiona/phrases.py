"""INTENT_PHRASES: the toy/original VCM dataset's canonical intent -> phrase
table (Option A's own content, mirroring `vcm.optionb.grammar`'s role for
Option B). See docs/VCM-CONTRACT.md section 3 for the full contract.
"""

from __future__ import annotations

# INTENT_PHRASES: <INTENT label> -> canonical normalized phrase.
# Verified against out/conversions/v2/reports/tmp-qa-<INTENT>-*.md "expected"
# column this session (see also the slow drift-guard test in
# tests/test_vcm_text.py that re-derives this table from those reports).
INTENT_PHRASES: dict[str, str] = {
    "ALARM": "set alarm",  # tmp-qa-ALARM-*.md
    "CALL": "call",  # tmp-qa-CALL-*.md
    "DIM_DOWN": "dimmer",  # tmp-qa-DIM-DOWN-*.md
    "DIM_UP": "brighter",  # tmp-qa-DIM-UP-*.md
    "LIGHT_OFF": "lights off",  # tmp-qa-LIGHT-OFF-*.md
    "LIGHT_ON": "lights on",  # tmp-qa-LIGHT-ON-*.md
    "LIST_REMINDERS": "list reminders",  # tmp-qa-LIST-REMINDERS-*.md
    "MESSAGE": "message",  # tmp-qa-MESSAGE-*.md
    "NEXT": "next",  # tmp-qa-NEXT-*.md
    "PAUSE": "pause",  # tmp-qa-PAUSE-*.md
    "PLAY_MUSIC": "play music",  # tmp-qa-PLAY-MUSIC-*.md
    "SET_REMINDER": "set reminder",  # tmp-qa-SET-REMINDER-*.md
    "STOP": "stop",  # tmp-qa-STOP-*.md
    "TEMP_DOWN": "cooler",  # tmp-qa-TEMP-DOWN-*.md
    "TEMP_UP": "warmer",  # tmp-qa-TEMP-UP-*.md
    "TIME": "time",  # tmp-qa-TIME-*.md
    "TIMER": "set timer",  # tmp-qa-TIMER-*.md
    "VOLUME_DOWN": "volume down",  # tmp-qa-VOLUME-DOWN-*.md
    "VOLUME_UP": "volume up",  # tmp-qa-VOLUME-UP-*.md
    "WEATHER": "weather",  # tmp-qa-WEATHER-*.md
}
