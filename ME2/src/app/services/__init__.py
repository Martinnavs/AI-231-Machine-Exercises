"""Device state services for the UI site (feature `ui-site`).

Each service is an explicit state machine patterned on the reference
`LightStateMachine` in `services/requirements.md`: an explicit
`(state, event) -> next_state` transition table plus a `dispatch` that applies
the transition if one is defined and **ignores** it otherwise (no exception,
no error -- the command was valid, the device simply had no transition for the
current state). Time-based services (timer/alarm/indicator/phone) additionally
expose a `tick()` advanced once per second by the shared app clock.

Services are pure in-process state: no I/O, no model imports, no audio. They
are constructed with an injectable `now` clock (for testability) where they
need the current time.
"""
