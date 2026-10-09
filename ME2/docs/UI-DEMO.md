# UI dashboard and headless demo

The UI (`src/app/`, FastAPI) is a separate service from the voice pipeline. The model side reaches it one way over HTTP: `python -m app.forward` reads the streaming runner's stdout JSONL and POSTs it to the UI. Spec: `feature-engineering/ui-site/SPEC.md`.

## Run it

Two terminals, from `ME2/` (on a Pi, `make` works for these targets because they call `.venv/bin` directly; set `APP_VENV_BIN=.venv-pi/bin` for the lean venv):

```bash
make app-lan                 # UI on 0.0.0.0:8000 (make app = 127.0.0.1 only; APP_HOST / APP_PORT override)
make app-pipeline-ctcwide    # hybrid CTC-wide model + wake word, piped into app.forward
```

`app-pipeline-ctcwide` uses the same settings as the streaming command in the README plus `--emit-listening`. `APP_MIC_COMMAND` overrides the mic, `APP_PIPELINE_SOURCE=clip.wav` replays a file. The UI has no authentication: anyone on the network can open it and POST `/api/command`, so use a trusted network.

`--emit-listening` makes the runner print `{"event":"listening","state":"active"|"passive"}` when the wake-word gate opens and closes. The UI's listening indicator and the music soft-pause follow it; without the flag stdout holds only trigger records ([`STREAMING-CONTRACT.md`](STREAMING-CONTRACT.md)).

## Music (for testing)

The UI's music panel plays real audio from `extras/music/` (git-ignored, so the audio never enters the repo). To try it, create the folder and add a few `.mp3`/`.wav`/`.flac`/`.ogg`/`.m4a` files:

```bash
mkdir -p extras/music && cp ~/Music/*.mp3 extras/music/    # then restart the UI: make app
```

File names become the track titles. With no files (or no `ffplay`) the panel stays silent and simulated. Playback starts at 30% volume; the wake word soft-pauses it and it resumes afterwards, except after a PAUSE or STOP command. Set `ME2_MUSIC_AUDIODEV` (an ALSA device, e.g. `plughw:CARD=CD002AUDIO,DEV=0`) to pick the speaker.


## Offline demo on a phone hotspot (Pi has no monitor)

The Pi runs headless: the UI and the live pipeline start as user services at boot, and the laptop shows the UI in a browser. `http://raspberrypi.local:8000` resolved from a laptop on the phone hotspot (checked); the UI needs no internet.

1. One-time, on the Pi: `./deploy/install-demo.sh` (enables `me2-ui` = `make app-lan` and `me2-pipeline` = `make app-pipeline-ctcwide`, plus linger so they start without a login). The mic is addressed by card name (`plughw:CARD=UACDemoV10,DEV=0`) because card numbers can change between boots; edit `deploy/systemd/me2-pipeline.service` for a different mic.
2. One-time: save the hotspot so the Pi auto-joins it:
   `nmcli device wifi connect "<ssid>" password "<pw>"`, then `nmcli connection modify "<ssid>" connection.autoconnect-priority 100`.
3. Demo: turn the hotspot on, power the Pi, join the laptop to the hotspot, open `http://raspberrypi.local:8000`. If the name does not resolve, use the Pi's IP (the phone's connected-devices list, or `nmap -sn <subnet>/24` from the laptop).
4. Debug: `journalctl --user -u me2-ui -u me2-pipeline -f`; restart with `systemctl --user restart me2-pipeline`. Run the services by hand only after `systemctl --user stop me2-ui me2-pipeline`, otherwise port 8000 and the mic are taken.

Not yet verified: a cold boot through the services, and the stable mic name capturing audio. The Pi has no battery clock: offline, its time after a cold boot is the last saved time, so alarm and timer demos that read the wall clock may be off. The UI has no authentication, so use a network you trust. Fallbacks if the hotspot is a problem (client isolation, `.local` not resolving): an Ethernet cable with a shared fixed address, or the Pi's own Wi-Fi access point.
