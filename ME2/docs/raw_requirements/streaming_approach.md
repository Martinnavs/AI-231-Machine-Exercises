Training and inferring on a fixed 1.0–1.5 second window is not only reasonable, it is the universal industry standard for edge keyword spotting.

Live streaming microphones do not record discrete 1.5-second files and wait; they stream continuous audio frames through a small rolling memory buffer without stalling the processor.

---

### How Streaming Actually Works (The Ring Buffer Pattern)

In production edge hosting, you never wait 1.5 seconds between predictions. The streaming pipeline operates decoupled into two parallel operations:

```
[ Microhpone ] ---> (Audio Callback: chunks of 50ms)
                           │
                           ▼
               [ Circular Ring Buffer (1.5s) ]
                           │
             (Sampled every 100ms stride)
                           ▼
             [ Mel-Spectrogram Extract (~1ms) ]
                           │
                           ▼
             [ Model Inference (~3ms on RPi) ]
                           │
                           ▼
             [ Posterior Smoothing / Debounce ]

```

* **Audio Capture Thread**: Reads small audio blocks (typically 512–1,024 samples, or 32–64 ms at 16 kHz) from ALSA/PyAudio directly into a fixed-size FIFO circular buffer holding exactly 1.5 seconds of PCM samples.
* **Inference Loop**: Runs at a fixed stride (e.g., every 100 ms). Every 100 ms, it takes the current 1.5-second snapshot from the buffer, computes the Mel spectrogram, and passes it to the quantized model.

---

### Compute & Latency Budget on Raspberry Pi 4/5

Running a 1.5s window at 10 Hz (every 100 ms) leaves ample CPU headroom:

| Operation | Typical Duration on RPi 4/5 | Duty Cycle (per 100 ms step) |
| --- | --- | --- |
| **Audio I/O (Chunk read)** | <0.1 ms (interrupt-driven) | Negligible |
| **Mel Spectrogram (1.5s)** | ~0.8–1.2 ms | ~1% CPU |
| **DS-CNN Inference (INT8)** | ~2.5–4.0 ms | ~3–4% CPU |
| **Total Work per Window** | **~3.5–5.5 ms** | **~4–6% of a single core** |

Because each inference takes under 5 ms, running it every 100 ms consumes less than 6% of one CPU core, leaving the other 3 cores completely untouched for your main application logic.

---

### Common Streaming Pitfalls & Fixes

* **Boundary Clipping (Split Commands)**:
* *Problem*: If a user starts speaking mid-window, the command is cut in half across two independent discrete blocks.
* *Fix*: The rolling stride solves this automatically. With a 100 ms stride over a 1.5-second window, a spoken phrase will be fully contained inside at least 3 to 6 overlapping inference windows.


* **Double-Triggering / Flickering**:
* *Problem*: Because the window slides by 100 ms, the model will output a high probability for "lights on" 3 or 4 consecutive times as the utterance passes through the buffer.
* *Fix*: Apply a **debounce / refractory state machine**: once a command crosses the confidence threshold (e.g., $p > 0.80$), lock detection and enforce a 1.0-second cooldown before allowing that command (or any command) to trigger again.


* **Audio Buffer Underrun**:
* *Problem*: Running model inference inside the audio driver's callback blocks the stream, causing dropped audio frames (ALSA buffer underruns).
* *Fix*: Decouple capture and inference. Let the sound callback only push raw bytes into a thread-safe `collections.deque(maxlen=24000)`, while a separate worker thread or process reads the buffer and executes inference.
