# Ambient-noise overlay -- vcm summary

Base manifest: `out/conversions/v2/user_voice/ambient_subset/manifest.csv` (953 rows). seed=0, p_mix=0.5, SNR uniform [0.0, 30.0] dB, 2 attempts per selected row, every stored mix ASR-gated (simple-audio-transcriber, v1-pair threshold 0.8 / wake-word trigger check for `_unknown_`).

## Gate results

- planned mixes: 634; stored: 515 (81.2% pass rate)
- attempt 1: 258/317 passed (81.4%)
- attempt 2: 257/317 passed (81.1%)
- per split (stored/planned): train 515/634

## Realized SNR histogram (stored mixes, 5 dB bins)

- [+0, +5) dB:    76 ############################################################################
- [+5, +10) dB:    70 ######################################################################
- [+10, +15) dB:    84 ####################################################################################
- [+15, +20) dB:   100 ####################################################################################################
- [+20, +25) dB:    94 ##############################################################################################
- [+25, +30) dB:    90 ##########################################################################################
- [+30, +30) dB:     1 #

## Clipping

- stored mixes clipped to ±1.0: 15/515 (2.9%)

## Chunk usage (source file -> stored mixes)

- 1 Hour Filipino Café⧸Coffee Shop Noise Ambience for Studying, Focus and Homework.wav: 158
- ANXIETY RELIEF ANTI-STRESS RELAXING AND CALMING MORNING SOUNDS OF SUBURBAN PHILIPPINES (1+ HOURS).wav: 192
- Boost productivity instantly with this 1 Hour Classroom Noises - Classroom Ambience - Study With Me.wav: 140
- COFFEE SHOP AMBIENCE ｜ People Talking ｜ FREE To Use.wav: 25

## License -- read before using this subset

The ambient noise mixed into this subset is Creative Commons licensed (YouTube source). The exact per-file CC variant is to be confirmed at attribution time; until then this subset is handled with the same isolation + note discipline as the ESC-50 (CC-BY-NC-SA-4.0) rows in this tree: it lives in its own subset/derived manifest, and every checkpoint trained on a manifest containing these rows must carry a license note naming this corpus (see out/conversions/v2/README.md's background_noise section for the precedent). Any file later determined to be CC-ND (no derivatives) must be excluded from the corpus and this subset rebuilt -- mixing is a derivative work.

Per-file attribution (fill at attribution time):

- 1 Hour Filipino Café⧸Coffee Shop Noise Ambience for Studying, Focus and Homework.wav: <uploader> -- <video URL> [CC variant unverified]
- ANXIETY RELIEF ANTI-STRESS RELAXING AND CALMING MORNING SOUNDS OF SUBURBAN PHILIPPINES (1+ HOURS).wav: <uploader> -- <video URL> [CC variant unverified]
- Boost productivity instantly with this 1 Hour Classroom Noises - Classroom Ambience - Study With Me.wav: <uploader> -- <video URL> [CC variant unverified]
- COFFEE SHOP AMBIENCE ｜ People Talking ｜ FREE To Use.wav: <uploader> -- <video URL> [CC variant unverified]
