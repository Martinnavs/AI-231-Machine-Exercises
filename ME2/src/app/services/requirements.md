Requirement: Create a fastapi-based site that does a lot of things.

Main architecture is a state class below:

from enum import Enum, auto

class State(Enum):
    OFF = 0
    ON_25 = 25
    ON_50 = 50
    ON_75 = 75
    ON_100 = 100

class Event(Enum):
    POWER_ON = auto()
    POWER_OFF = auto()
    INCREASE = auto()
    DECREASE = auto()

class LightStateMachine:
    # Explicit mapping: (Current_State, Event) -> Next_State
    TRANSITIONS = {
        # From OFF: power on defaults to 25% brightness
        (State.OFF, Event.POWER_ON): State.ON_25,
        
        # From any active state: power off
        (State.ON_25, Event.POWER_OFF): State.OFF,
        (State.ON_50, Event.POWER_OFF): State.OFF,
        (State.ON_75, Event.POWER_OFF): State.OFF,
        (State.ON_100, Event.POWER_OFF): State.OFF,

        # Increasing brightness (capped at 100%)
        (State.ON_25, Event.INCREASE): State.ON_50,
        (State.ON_50, Event.INCREASE): State.ON_75,
        (State.ON_75, Event.INCREASE): State.ON_100,
        (State.ON_100, Event.INCREASE): State.ON_100,  # Caps at 100%

        # Decreasing brightness (capped at 25%)
        (State.ON_100, Event.DECREASE): State.ON_75,
        (State.ON_75, Event.DECREASE): State.ON_50,
        (State.ON_50, Event.DECREASE): State.ON_25,
        (State.ON_25, Event.DECREASE): State.ON_25,    # Caps at 25%
    }

    def __init__(self):
        self.state = State.OFF

    def dispatch(self, event: Event):
        next_state = self.TRANSITIONS.get((self.state, event))
        if next_state is not None:
            self.state = next_state
            print(f"[{event.name}] -> {self.state.name} ({self.state.value}%)")
        else:
            print(f"Ignored: Cannot perform '{event.name}' while in state '{self.state.name}'")

UI and Logic below:
Music player - Bottom-center UI - play/pause, next, stop, and volume bar
Lights bulb - Top right UI - Light On/Lights Off, Color (red, green, blue), Brightness (0, 20, 60, 100%)
Phone - Bottom-right UI - Call (random call to a random person), Message (send a message to a random person)
Reminder List - Top left UI - Add reminder (add it via <Slot> - <timestamp>), list reminders (scrollable list of reminders) with exit icon afterwards, default view is the latest reminder
Thermostat - Top-center UI - set temperature 
Timer - Bottom left UI - MM:SS
Indicator UI - Passive listening icon (while waiting for wakeword), Active listening icon with semicircle for 3 seconds (waiting for VCM), bottom of icon would be the detected word, and bottom it would be for time and weather a la -- Current time is <> or Weather is <>.

