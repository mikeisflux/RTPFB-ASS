# Streaming the Avatar into OBS → Twitch / YouTube / Zoom

The pipeline already produces an OBS-ready feed: video goes out as a virtual
camera, audio goes out as a virtual microphone. OBS picks both up, syncs them,
and pushes to your destination. Below is the working setup.

## 1. One-time system setup

### Loopback audio device (so OBS can hear the converted voice)

| OS | Setup |
|---|---|
| Linux | `pactl load-module module-null-sink sink_name=rtpfb_out sink_properties=device.description=rtpfb_out` |
| Windows | Install [VB-CABLE](https://vb-audio.com/Cable/). Pipeline writes to `CABLE Input`, OBS reads `CABLE Output`. |
| macOS | Install [BlackHole](https://existential.audio/blackhole/) (2ch). Pipeline writes to `BlackHole 2ch`, OBS reads the same device. |

### Virtual camera driver

| OS | Setup |
|---|---|
| Linux | `sudo modprobe v4l2loopback devices=1 card_label="rtpfb-cam" exclusive_caps=1` |
| Windows | OBS Studio installs the OBS Virtual Camera driver automatically. |
| macOS | OBS installs its virtual camera; `pyvirtualcam` writes to it. |

## 2. Run the pipeline

```bash
# clone a voice (1–5 min of audio is plenty for KNN-VC)
rtpfb voice add alice --samples ~/voices/alice/ --method knn-vc

# verify
rtpfb voice list

# go live
rtpfb run \
    --target ./assets/alice_face.png \
    --voice --voice-model alice \
    --virtual-mic rtpfb_out
```

## 3. OBS sources

In OBS:

* **Video Capture Device** → `OBS Virtual Camera` (Win/macOS) or `/dev/video<N>` matching the v4l2loopback card (Linux). Set resolution/FPS to match `--width/--height/--fps`.
* **Audio Input Capture** → `Monitor of rtpfb_out` (Linux PulseAudio), `CABLE Output (VB-Audio Virtual Cable)` (Windows), or `BlackHole 2ch` (macOS).
* **Mute** the OBS desktop / mic source you don't want; the converted-voice mic is the only audio you should be sending.

Stream → Settings → Service: Twitch / YouTube / Custom RTMP. Done.

## 4. A/V sync

OBS handles A/V sync with its own buffer, but our two streams are independent. If lips lead or lag the audio:

* In OBS, right-click the **audio source** → `Sync Offset` → adjust by 50ms steps.
* Typical offsets for this pipeline: **+100 to +250 ms** (audio later than video) because Wav2Lip + render is slightly faster than the audio thread's output buffer.

## 5. Latency expectations

| Hop | Latency |
|---|---|
| Webcam capture → frame ready | ~16–33 ms |
| Face swap (InsightFace) | ~20–40 ms on a 5090 |
| Stabilize + virtual cam send | ~5–10 ms |
| **Video glass-to-glass to OBS** | **~50–100 ms** |
| Mic capture (256 sample block @ 16 kHz) | 16 ms |
| Voice changer (KNN-VC / RVC) | 30–80 ms |
| Virtual mic block out | 16 ms |
| **Audio glass-to-OBS** | **~62–112 ms** |
| OBS encode + RTMP | 1–3 s (Twitch low-latency mode) |
| Twitch CDN to viewer | 2–4 s |

The 115 ms voice-conversion target is **on the audio path before OBS** — Twitch's own buffer is the dominant viewer-side delay.

## 6. Going lower latency than virtual cam

Virtual camera adds a frame copy and runs in user space. If you need sub-frame video latency to OBS:

* **NDI** — install [obs-ndi](https://github.com/obs-ndi/obs-ndi) and switch us to NDI output. Gives frame-accurate timestamps so OBS A/V sync is automatic. (Not yet wired — open issue.)
* **Direct RTMP** — skip OBS entirely and have us push to Twitch's RTMP ingest via ffmpeg. Loses OBS scenes / overlays.
