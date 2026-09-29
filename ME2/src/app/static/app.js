/* UI site dashboard (feature `ui-site`) — vanilla JS, no build step.
 *
 * The page is a pure receiver: it renders whatever the server state says.
 * Local clicks POST to the REST actions (snappy local feedback from the
 * response) and the WebSocket re-renders on every server-side push
 * (model commands, the 1-second tick, other browsers). */

"use strict";

const $ = (id) => document.getElementById(id);

let state = null;
let vocab = null;
let ws = null;
let lastListening = "passive";

const COLOR_HEX = {
  white: "#f2f2f2",
  red: "#e5484d",
  green: "#46a758",
  blue: "#3e8bfd",
};

/* --- bootstrap ---------------------------------------------------------- */

async function init() {
  vocab = await (await fetch("/api/vocab")).json();
  fillSelect("reminder-task", vocab.reminder_tasks);
  fillSelect("alarm-time", vocab.alarm_times);
  bindActions();
  state = await (await fetch("/api/state")).json();
  render(state);
  connectWs();
}

function fillSelect(id, values) {
  const sel = $(id);
  sel.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }
}

function bindActions() {
  document.querySelectorAll("[data-act]").forEach((btn) => {
    btn.addEventListener("click", () => onAction(btn.dataset.act, btn));
  });
}

/* --- REST actions (local clicks) ------------------------------------------ */

async function onAction(act, btn) {
  let path;
  let body;
  switch (act) {
    case "light-power-on": path = "/api/light/power_on"; break;
    case "light-power-off": path = "/api/light/power_off"; break;
    case "light-increase": path = "/api/light/increase"; break;
    case "light-decrease": path = "/api/light/decrease"; break;
    case "light-color": path = "/api/light/set_color"; body = { color: btn.dataset.color }; break;
    case "light-level": path = "/api/light/set_brightness"; body = { level: Number(btn.dataset.level) }; break;
    case "thermo-set": path = "/api/thermostat"; body = { degrees: Number(btn.dataset.degrees) }; break;
    case "alarm-set": path = "/api/alarm/set"; body = { time: $("alarm-time").value }; break;
    case "alarm-reset": path = "/api/alarm/reset"; break;
    case "timer-start": path = "/api/timer/start"; body = { duration_s: Number(btn.dataset.duration) }; break;
    case "timer-reset": path = "/api/timer/reset"; break;
    case "music-play": path = "/api/music/play"; break;
    case "music-pause": path = "/api/music/pause"; break;
    case "music-next": path = "/api/music/next"; break;
    case "music-stop": path = "/api/music/stop"; break;
    case "music-vol-up": path = "/api/music/volume_up"; break;
    case "music-vol-down": path = "/api/music/volume_down"; break;
    case "phone-call": path = "/api/phone/call"; break;
    case "phone-hang-up": path = "/api/phone/hang_up"; break;
    case "phone-message": path = "/api/phone/message"; break;
    case "reminder-add": path = "/api/reminders"; body = { task: $("reminder-task").value }; break;
    case "reminders-expand": path = "/api/reminders/expand"; break;
    case "reminders-collapse": path = "/api/reminders/collapse"; break;
    default: return;
  }
  const opts = { method: "POST" };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  try {
    const res = await fetch(path, opts);
    const data = await res.json();
    if (data.state) {
      state = data.state;
      render(state);
    }
  } catch (err) {
    console.error("action failed:", act, err);
  }
}

/* --- websocket (server pushes) --------------------------------------------- */

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (msg) => {
    state = JSON.parse(msg.data);
    render(state);
  };
  ws.onclose = () => setTimeout(connectWs, 1500);
  ws.onerror = () => {
    try { ws.close(); } catch (_) { /* already closed */ }
  };
}

/* --- render ------------------------------------------------------------------- */

function render(s) {
  renderIndicator(s.indicator);
  renderReminders(s.reminders);
  renderThermostat(s.thermostat);
  renderAlarm(s.alarm);
  renderLights(s.lights);
  renderTimer(s.timer);
  renderMusic(s.music);
  renderPhone(s.phone);
}

function renderIndicator(ind) {
  const active = ind.listening === "active";
  $("listening-dot").classList.toggle("active", active);
  $("listening-label").textContent = active
    ? "active — waiting for command"
    : "passive — waiting for wakeword";

  const sweep = $("sweep");
  if (active && lastListening !== "active") {
    // (Re)start the 3-second semicircle each time the gate opens.
    sweep.classList.remove("sweeping");
    void sweep.getBoundingClientRect(); // force reflow so the animation restarts
    sweep.classList.add("sweeping");
  } else if (!active) {
    sweep.classList.remove("sweeping");
  }
  lastListening = ind.listening;

  $("detected").textContent = ind.detected ? `Heard: ${ind.detected}` : "";
  const statusEl = $("status-line");
  statusEl.textContent = ind.status_line || "";
  statusEl.classList.toggle("hidden", !ind.status_line);
}

function renderReminders(r) {
  const latest = r.latest;
  $("reminder-latest").textContent = latest
    ? `Latest: ${latest.task} — ${latest.logged_at}`
    : "No reminders yet.";
  const list = $("reminder-list");
  list.classList.toggle("hidden", !r.expanded);
  list.innerHTML = "";
  if (r.expanded) {
    for (const item of r.items) {
      const li = document.createElement("li");
      li.textContent = `${item.logged_at} · ${item.task}`;
      list.appendChild(li);
    }
  }
  $("btn-reminders-collapse").classList.toggle("hidden", !r.expanded);
  $("btn-reminders-expand").disabled = r.expanded;
}

function renderThermostat(t) {
  $("thermo-value").textContent = `${t.degrees}°`;
  document.querySelectorAll("[data-act='thermo-set']").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.degrees) === t.degrees);
  });
}

function renderAlarm(a) {
  const el = $("alarm-readout");
  if (a.state === "DUE") {
    el.textContent = `ALARM DUE — ${a.display}`;
    el.classList.add("due");
  } else if (a.state === "SET") {
    el.textContent = `Alarm set for: ${a.display}`;
    el.classList.remove("due");
  } else {
    el.textContent = "No alarm set";
    el.classList.remove("due");
  }
  $("btn-alarm-reset").disabled = a.state === "UNSET";
}

function renderLights(l) {
  $("lights-status").textContent = l.on ? `On at ${l.brightness}%` : "Off";
  $("lights-dot").style.background = l.on ? COLOR_HEX[l.color] : "#555";
  document.querySelectorAll("#light-levels .level-seg").forEach((seg) => {
    seg.classList.toggle("lit", l.on && l.brightness >= Number(seg.dataset.level));
  });
  document.querySelectorAll("[data-act='light-level']").forEach((b) => {
    b.classList.toggle("active", l.on && l.brightness === Number(b.dataset.level));
  });
  document.querySelectorAll("[data-act='light-color']").forEach((b) => {
    b.classList.toggle("active", l.color === b.dataset.color);
  });
}

function renderTimer(t) {
  $("timer-value").textContent = t.display;
  const stateEl = $("timer-state");
  stateEl.textContent =
    t.state === "RUNNING" ? "Timer running"
    : t.state === "DONE" ? "Timer done"
    : "Timer idle";
  stateEl.classList.toggle("done", t.state === "DONE");
  document.querySelectorAll("[data-act='timer-start']").forEach((b) => {
    b.disabled = t.state === "RUNNING"; // re-issue is ignored; wait it out
  });
  $("btn-timer-reset").disabled = t.state === "IDLE";
}

function renderMusic(m) {
  let line;
  if (m.state === "PLAYING") {
    line = `▶ Track ${m.track_index + 1}/${m.track_total} — ${m.track}`;
  } else if (m.state === "PAUSED") {
    line = `❚❚ Track ${m.track_index + 1}/${m.track_total} — ${m.track}`;
  } else {
    line = "Stopped";
  }
  $("music-track").textContent = line;
  $("music-volume").textContent = `${m.volume}%`;
  $("music-volume-bar").style.width = `${m.volume}%`;
}

function renderPhone(p) {
  const line = $("phone-status");
  if (p.state === "IN_CALL" && p.person) {
    line.textContent =
      `On call with ${p.person.name} (${p.person.number}) — ${fmtMMSS(p.elapsed_s)}`;
    $("btn-phone-hang-up").classList.remove("hidden");
  } else {
    line.textContent = "Not on a call";
    $("btn-phone-hang-up").classList.add("hidden");
  }
  const log = $("phone-messages");
  log.innerHTML = "";
  for (const m of p.messages.slice(0, 12)) {
    const li = document.createElement("li");
    li.textContent = `${m.at} · ${m.name}: ${m.text}`;
    log.appendChild(li);
  }
  if (p.messages.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No messages yet.";
    log.appendChild(li);
  }
}

function fmtMMSS(total) {
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

init();
