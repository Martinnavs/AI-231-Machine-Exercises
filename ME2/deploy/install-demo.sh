#!/usr/bin/env bash
# Install the demo as user systemd services that start at boot (no sudo
# needed except what `loginctl enable-linger` may ask for).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p ~/.config/systemd/user
cp "$here"/systemd/me2-ui.service "$here"/systemd/me2-pipeline.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable me2-ui.service me2-pipeline.service
loginctl enable-linger "$USER"   # run user services at boot without a login
echo "Installed. Start now: systemctl --user start me2-ui me2-pipeline  (or reboot)"
echo "Logs: journalctl --user -u me2-ui -u me2-pipeline -f"
