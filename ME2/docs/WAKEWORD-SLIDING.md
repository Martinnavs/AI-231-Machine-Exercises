# Wake word + sliding window: answer as soon as the command ends

Date: 2026-10-01. Plan and decisions: `.scratch/wakeword-sliding/PLAN.md`. Contract:
`docs/STREAMING-CONTRACT.md` ("The endpointed policy").

## Why

The previous serving mode (`--policy single_period --gate-period 3`) waits for a fixed 3 s period after the
wake word and decodes once at its end. The course requirement is to return the result as soon as the spoken
command finishes, without a fixed listening window.

## What changed

New policy `--policy endpointed` (`EndpointedPeriodPolicy`). The wake word still opens a period. Every 0.25 s
the decoder runs on the audio since the wake word (growing to 2.5 s, then sliding), and the first decode that
is confident (threshold and incomplete-prefix margin, as before) and ended (the last `--hold-ms` of model
output is blank with probability >= 0.9) is emitted at once and closes the period. If nothing qualifies
within 3 s of the last wake-word detection, the period closes with no output.

Run it with `make app-pipeline-live` (QuartzNet INT8, the settings below). `make app-pipeline` is unchanged.

## How it was measured

Replay sessions built from real audio (`vcm.streaming.session_replay`), one per test command:
room tone throughout, a real "sesame" wake-word clip, a 0.1-0.4 s gap, the command, and 3.5 s of trailing
room tone. Commands follow the earlier streaming recall check: every PAUSE, STOP and TIME clip of the split,
plus the first 30 of each other intent (807 val, 821 test sessions). The end of speech comes from forced
alignment of the clean command with the stride-1 production model, so latency is measured from when the
speaker stopped.

`vcm.streaming.replay_eval` runs the real streaming CLI on each session file. File replay runs in lockstep,
so the latency figures are algorithmic. Add one decode's compute (about 135 ms for QuartzNet INT8 on the
EPYC server) for wall-clock latency.

Fixed for every run: QuartzNet stride 2 INT8 (`out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m`), threshold
-0.1, margin 4.0, mean-frame scoring, beam 50, wake word `out/wakeword-sesame-ambient-rir-45m`.

## Choosing the setting (validation, rule fixed before results)

Rule: most correct first triggers among settings with no more wrong-intent triggers than the 3 s baseline;
among settings within 0.5 pp of that, the lowest p95 latency.

| Val (807 sessions) | Correct first trigger | Wrong-intent triggers | Missed | Latency p50 / p95 |
|---|---|---|---|---|
| 3 s mode (baseline) | 496 (61.5%) | 27 | 284 | 2.19 / 2.69 s |
| hold 200 ms, stable 1 (chosen) | 756 (93.7%) | 14 | 40 | 0.36 / 0.50 s |
| hold 200 ms, stable 2 | 755 | 8 | 45 | 0.37 / 0.51 s |
| hold 300 ms, stable 1 | 748 | 14 | 48 | 0.46 / 0.62 s |
| hold 300 ms, stable 2 | 745 | 8 | 55 | 0.46 / 0.62 s |
| hold 400 ms, stable 1 | 741 | 14 | 55 | 0.57 / 0.70 s |
| hold 400 ms, stable 2 | 739 | 8 | 61 | 0.57 / 0.70 s |

Requiring two identical decodes in a row ("stable 2") roughly halves wrong-intent triggers (14 to 8) for
about one fewer correct session. The rule chose stable 1; stable 2 is the safer alternative if wrong actions
matter more than a 0.1 pp accuracy difference (`APP_LIVE_STABLE=2`).

## Result (test, run once with the chosen setting)

| Test (821 sessions) | 3 s mode | Endpointed |
|---|---|---|
| Correct first trigger | 519 (63.2%) | 741 (90.3%) |
| Wrong-intent triggers | 27 | 21 |
| Missed | 275 | 59 |
| Latency from end of speech, p50 / p95 | 2.17 / 2.67 s | 0.36 / 0.51 s |
| Beam-search decodes per session | 0.93 | 7.1 |
| PAUSE / STOP / TIME correct | 66/100, 72/134, 62/107 | 93/100, 122/134, 101/107 |

All success criteria set in the plan that this replay can measure pass: accuracy and wrong intents no worse
than the 3 s mode, and latency p50 <= 0.5 s and p95 <= 0.9 s.

## Incomplete-prefix margin in streaming mode

The whole-clip recalibration on the `quartznet-promotion` branch (`docs/archive/QUARTZNET-OPERATING-POINT.md`)
picked margin 20.0 at threshold -0.01 for the 3 s mode. This mode was swept separately on val (hold 200 ms,
stable 1, threshold -0.1), with the rule fixed beforehand: most correct first triggers among margins with no
more wrong-intent triggers than the 3 s baseline, then fewest wrong-intent triggers within 0.5 pp, then p95.

| Val (807 sessions) | Correct first trigger | Wrong-intent triggers | Missed | TIME correct |
|---|---|---|---|---|
| **margin 4.0 (chosen)** | 756 (93.7%) | 14 | 40 | 106/109 |
| margin 10 | 744 (92.2%) | 5 | 60 | 87/109 |
| margin 15 | 732 (90.7%) | 5 | 71 | 76/109 |
| margin 20 | 731 (90.6%) | 4 | 73 | 75/109 |

Latency is unchanged across margins (p50 0.36 s, p95 0.50 s). Larger margins mostly reject genuine TIME
commands: "time" is a character prefix of "timer ...", and in growing windows the gate treats a real "time"
as a possibly unfinished "timer". The whole-clip calibration set has too few TIME clips to show this. Margin
10 cuts wrong-intent triggers to about a third at the cost of 12 correct sessions, mostly TIME; it is the
alternative if wrong actions matter more (`APP_LIVE_MARGIN=10`). The test numbers above are at margin 4.0.

The two serving modes therefore use different margins if the `quartznet-promotion` branch is merged:
`app-pipeline-quartznet` (3 s mode, whole-window decode) 20.0, `app-pipeline-live` (this mode) 4.0.

## Reading the numbers

- The 3 s mode does much worse here than whole-clip evaluation (98-99%). Its 3 s window starts at the wake
  word gate's *latest* detection, which can be about a second after "sesame" ends, so on longer commands the
  window is mostly room tone or misses part of the command. The endpointed window starts at the *first*
  detection. This explanation is consistent with the numbers but was not isolated by an experiment.
- In 43 of the 59 endpointed misses on test, no period opened at all: the wake-word model did not detect
  "sesame" in that session. No serving policy can recover those.
- Wake-word clips come from the wake-word dataset (many are voice-converted) and the command clips from the
  VCM test set, joined with a short gap. Real users may pause differently.

## Not measured

- Realtime runs (dropped windows). Decoding about 7 times per command instead of once costs compute; at
  about 135 ms per decode QuartzNet fits the 250 ms stride on the server, the 1.01M MatchboxNet (about
  246 ms) would not. No Raspberry Pi has been measured.
- False actions on long real ambient audio with this policy (the cascade soak) and background speech near the
  speaker, which can delay the end-of-speech check until the time-out.
- The threshold was not re-tuned for growing windows (the margin was, see above); it is the same as the
  baseline's.

## Next step (proposed)

Endpoint first, then decode: run only the cheap model forward each stride, use its blank probabilities to
detect the end of speech, and run the beam search once at that point. Expected: the same latency with about
one beam search per command instead of seven.

## Reproduce

```bash
uv run python -m me2_voicegen.vcm.streaming.session_replay --split val  --out-dir out/vcm/wakeword-sliding/sessions-val
uv run python -m me2_voicegen.vcm.streaming.session_replay --split test --out-dir out/vcm/wakeword-sliding/sessions-test
uv run python -m me2_voicegen.vcm.streaming.replay_eval --sessions out/vcm/wakeword-sliding/sessions-test \
  --model out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m --threshold -0.1 --margin 4.0 \
  --policy endpointed --hold-ms 200 --stable-strides 1 --out out/vcm/wakeword-sliding/test/endpointed_h200_s1.json
```

The sessions used here were first built with a 1.5 s tail and then extended in place to 3.5 s by repeating
the final room tone (the 1.5 s tail ended sessions before the 3 s mode's period closed, unfairly penalising
it). The builder now uses 3.5 s directly, so a fresh build differs only in that tail's room-tone content.
