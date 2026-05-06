# Deployment & First-Run Guide

End-to-end walkthrough from a clean machine to streaming a photoreal female
avatar with soft-body chest physics and a cloned voice. Everything in this
guide is free.

**Plan ~6 hours total**, mostly waiting on Unreal + MetaHuman downloads.

---

## What you need

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 (primary) or Linux | Windows 11 |
| GPU | NVIDIA RTX 3080 (10 GB VRAM) | RTX 4090 / 5090 (24 GB VRAM) |
| RAM | 32 GB | 64 GB |
| Disk free | ~120 GB | 200 GB |
| CUDA | 12.x | 12.4+ |
| Webcam + microphone | required | required |

This guide is Windows-primary. Linux notes are inline where they differ.

---

## Phase 1 — Python pipeline (~30 min)

### 1.1 Install Python 3.11

Windows: download from [python.org](https://www.python.org/downloads/) and tick **"Add to PATH"** during install.

Linux:
```bash
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### 1.2 Clone the repo

```bash
git clone https://github.com/mikeisflux/RTPFB-ASS.git
cd RTPFB-ASS
git checkout claude/add-initial-setup-files-b1uaU
```

### 1.3 Create a virtualenv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 1.4 Install the package

```bash
pip install --upgrade pip
pip install -e ".[voice,mocap,preset,lipsync]"
```

This pulls numpy, opencv, mediapipe, insightface, onnxruntime-gpu, sounddevice, pyvirtualcam, torch, librosa, python-osc, websocket-client, and a few others. It'll take a few minutes.

### 1.5 Pull upstream repos

```bash
bash scripts/install_third_party.sh
```

Clones voice-changer, RVC-WebUI, knn-vc, Wav2Lip, and others into `third_party/` (gitignored). Takes ~5 min depending on network.

### 1.6 Download model files

```bash
bash scripts/download_models.sh
```

This auto-fetches `face_landmarker.task` (~3 MB, MediaPipe Apache-2.0). Other model weights have terms of use — the script prints what to grab and where.

You need at minimum:
- `models/face_landmarker.task` (auto-fetched above)

### 1.7 Verify

```bash
rtpfb --help
rtpfb voice list
```

Should print the help banner and `(no voices registered — use `rtpfb voice add`)`.

---

## Phase 2 — Voice cloning (~20 min)

### 2.1 Record samples

Record 1–5 minutes of clean speech in your **target voice** (the voice you want to *sound like*, not your own). Sources:

- A friend who'll lend their voice
- A YouTube clip of someone speaking clearly (extract the audio with ffmpeg/yt-dlp)
- Any clean speech: podcast, audiobook, voice memos

Save as `.wav`, `.mp3`, `.flac`, or `.m4a` in a folder, e.g. `~/voice_samples/alice/`. The cleaner and more varied, the better the clone.

### 2.2 Clone the voice

```bash
rtpfb voice add alice --samples ~/voice_samples/alice/ --method knn-vc --accent en-US
rtpfb voice list
```

`knn-vc` is zero-shot — no training run, just caches WavLM features. First call downloads WavLM (~1 GB) into `~/.cache/torch/hub/`.

### 2.3 Loopback audio device

This is the virtual mic OBS reads from.

**Windows:** install [VB-CABLE](https://vb-audio.com/Cable/). Pipeline writes to `CABLE Input`, OBS reads `CABLE Output`.

**Linux:**
```bash
pactl load-module module-null-sink \
  sink_name=rtpfb_out \
  sink_properties=device.description=rtpfb_out
```

**macOS:** install [BlackHole 2ch](https://existential.audio/blackhole/).

### 2.4 Smoke-test the voice path (optional)

```bash
rtpfb run \
  --preset presets/voice-only.yaml \
  --target placeholder.png \
  --voice-model alice \
  --virtual-mic "CABLE Input"  # or rtpfb_out / BlackHole 2ch
```

Speak — your voice comes out the loopback in the cloned voice. `Ctrl-C` to stop. (Skip this if you want to go straight to the full pipeline.)

---

## Phase 3 — Unreal Engine 5 (~2 hours, mostly downloads)

### 3.1 Epic Games account + launcher

1. Make a free Epic account at [epicgames.com](https://www.epicgames.com/account/).
2. Install the **Epic Games Launcher**.

### 3.2 Install Unreal Engine 5.4+

In the launcher: **Unreal Engine** tab → **Library** → **+** next to "Engine Versions" → pick **5.4** (or later) → **Install**. ~50 GB.

### 3.3 New project

When the engine finishes installing:
1. **Launch** → **Games** → **Blank** template
2. **Blueprint** (not C++)
3. **Maximum Quality**, **Starter Content: off**, **Raytracing: on**
4. Project name: `RTPFB-Avatar` (or whatever)
5. **Create**

Wait for it to compile shaders.

### 3.4 Add a MetaHuman

1. **Window** → **Quixel Bridge** → log in with your Epic account.
2. **MetaHumans** tab. Either:
   - Pick a stock **female MetaHuman** and click **Download** (Highest Quality).
   - Or click **MetaHuman Creator** to design one in the browser. Free.
3. Once downloaded, drag the MetaHuman into your level (the viewport).
4. **File → Save All**.

You should now see a photoreal woman standing in your level.

---

## Phase 4 — VMC → MetaHuman bridge (~30 min)

### 4.1 Install EVMC4U plugin

Free plugin that receives our VMC packets and drives a skeletal mesh.

```bash
cd <your-unreal-project>/
mkdir -p Plugins
cd Plugins
git clone https://github.com/HAL9HARUKU/EVMC4U.git
```

Restart Unreal. **Edit → Plugins** → search "EVMC4U" → tick the **Enabled** box. Restart Unreal again when prompted.

### 4.2 Add the EVMC4U receiver

1. **Content Drawer** → right-click → **Blueprint Class** → **Actor**. Name it `BP_VMCReceiver`.
2. Open it. **+ Add Component** → **EVMC4U Streaming Skeletal Mesh Component** (or the equivalent in your EVMC4U fork's component list).
3. In the component **Details**:
   - **Port**: `39539`
   - **Target Skeletal Mesh**: drag your MetaHuman's body skeletal mesh asset here.
   - **Bone Map**: pick the **VRM → MetaHuman** preset if available, or hand-map (see EVMC4U wiki for the table).
4. Drag `BP_VMCReceiver` into your level.

### 4.3 First sync test

In one terminal (with venv active):
```bash
rtpfb run --preset presets/mocap-unreal.yaml --no-preflight
```
(`--no-preflight` skips the camera check until we wire OBS.)

In Unreal, hit **Play** in the level. The MetaHuman should start mirroring your motion. If it twitches once and freezes, the bone map is wrong — fix the EVMC4U bone mapping table.

---

## Phase 5 — Soft-body physics (~30 min)

This is what gives you squish-on-lean.

### 5.1 Open the body cloth editor

1. In **Content Drawer**, find your MetaHuman's body skeletal mesh asset (something like `f_med_nrw_body`).
2. Double-click → **Skeletal Mesh** editor opens.
3. **Window → Cloth** to open the cloth panel.

### 5.2 Mask the chest as cloth-driven

1. Select the cloth painting tool.
2. Paint the chest region with `MaxDistance = 5–10` (cloth-driven).
3. Paint everywhere else with `MaxDistance = 0` (kinematic, follows the bone exactly).
4. Adjust `BackstopRadius` and `BackstopDistance` so the cloth doesn't clip into the ribs.

### 5.3 Tune the simulation

In the cloth asset's **Properties**:
- **EdgeStiffness**: 0.7–0.9 (firmness)
- **BendingStiffness**: 0.3–0.5 (squish softness)
- **Damping**: 0.3 (settles after motion)
- **Gravity Scale**: 1.0

Save, hit **Simulate** in the editor and tilt the bone preview to test. Once it looks right, save the asset.

(Optional, UE 5.4+: switch to **Chaos Flesh** for actual volumetric soft body. Heavier but more realistic compression.)

---

## Phase 6 — OBS + NDI (~20 min)

### 6.1 Install OBS

[obsproject.com](https://obsproject.com/). Free.

### 6.2 NDI plugins

OBS side: install [obs-ndi](https://github.com/obs-ndi/obs-ndi/releases).

Unreal side: install the [Unreal NDI plugin](https://github.com/ufna/UnrealNDI) into `<project>/Plugins/`.

Restart both. Enable both plugins in their respective settings.

### 6.3 Configure Unreal NDI output

In Unreal:
1. **Edit → Plugins → NDI IO** enabled.
2. Add an **NDI Broadcast Sender** to your level (or use the **Cinecam → NDI** output from your level's primary camera).
3. Frame the camera on the MetaHuman, set output resolution to match what you want to stream (1280×720 is fine).

### 6.4 OBS sources

In OBS, **Scenes → +** for a new scene, then **Sources → +**:

1. **NDI Source** → pick the Unreal sender. **Bandwidth: Highest. Sync: Source Timing.**
2. **Audio Input Capture** → pick the loopback monitor:
   - Windows: `CABLE Output (VB-Audio Virtual Cable)`
   - Linux: `Monitor of rtpfb_out`
   - macOS: `BlackHole 2ch`
3. (Optional) **Mute** OBS's default mic source so only the converted voice goes out.

### 6.5 Configure streaming

OBS → **Settings → Stream**:
- **Service**: Twitch (or YouTube / Custom RTMP)
- **Server**: closest ingest
- **Stream Key**: paste from Twitch dashboard

**Settings → Output**:
- **Output Mode**: Advanced
- **Encoder**: NVENC H.264 (or HEVC if your Twitch tier allows)
- **Bitrate**: 6000 kbps (Twitch cap) / 8000+ for YouTube
- **Keyframe Interval**: 2

---

## Phase 7 — Going live (~5 min, every time)

Order matters slightly: start the renderer first so the EVMC4U receiver is listening when our pipeline starts shipping packets.

### 7.1 Start Unreal

1. Open your project.
2. Hit **Play** (Alt-P) in the level. EVMC4U starts listening on port 39539.

### 7.2 Start the Python pipeline

```bash
cd RTPFB-ASS
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

rtpfb run --preset presets/mocap-unreal.yaml --voice-model alice
```

You should see in the log:
```
[INFO] rtpfb.health:  ok camera — index=0
[INFO] rtpfb.health:  ok face-landmarker[face_landmarker.task]
[INFO] rtpfb.vmc: VMC sender → 127.0.0.1:39539
[INFO] rtpfb.pipeline: pipeline starting (mode=mocap)
```

The MetaHuman should now mirror your motion in Unreal in real time.

### 7.3 Start streaming in OBS

1. Confirm the NDI source shows the MetaHuman.
2. Confirm the audio meter on your loopback source jumps when you speak.
3. **Start Streaming**.

### 7.4 A/V sync

Lips usually lead the audio slightly. In OBS, right-click the audio source → **Advanced Audio Properties** → **Sync Offset**. Typical setting for this pipeline: **+100 to +250 ms**.

---

## Phase 8 — Stop / restart cycle

### Stop
- `Ctrl-C` in the rtpfb terminal (graceful shutdown).
- Stop Play in Unreal.
- Stop streaming in OBS.

### Re-run after a config change
Just `rtpfb run --preset ...` again. Unreal can stay open.

### Switch voice
```bash
rtpfb voice list
rtpfb run --preset presets/mocap-unreal.yaml --voice-model bob
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `preflight failed: camera index 0 not openable` | Pass `--camera 1` (or higher) to find your webcam. |
| `face_landmarker.task missing` | Re-run `bash scripts/download_models.sh`. |
| `pythonosc not installed` | `pip install -e ".[mocap]"` |
| MetaHuman doesn't move | EVMC4U bone map is wrong. Check the receiver's bone-name mapping; verify port 39539 isn't firewalled. |
| Avatar twitches violently | Coordinate sign issue — toggle EVMC4U's "Mirror" or "Y-Up" flags. |
| Voice cuts in/out | Audio loopback not selected; check `--virtual-mic` matches the OS device name exactly. |
| Lips out of sync with mouth | Lower `--audio-blocksize` (try 128) and adjust OBS Sync Offset. |
| FPS is bad in Unreal | Lower the level's quality settings, drop NDI to 1280×720, disable Lumen if needed. |
| `rtpfb` command not found | Activate the venv (`source .venv/bin/activate` / `.venv\Scripts\activate`) and confirm `pip install -e .` succeeded. |

### Health check shortcut

If you're not sure why something's failing:

```bash
rtpfb health --preset presets/mocap-unreal.yaml --voice-model alice
```

It runs every preflight check and prints `ok`/`FAIL` for each — camera, GPU, model files, third-party repos, voice library entry, etc.

---

## What's not in this guide

- **Hand-tuning the MetaHuman's clothing / hair physics** — that's in the body's Cloth asset alongside the chest. Same workflow as Phase 5.
- **Live Link Face for higher-quality facial animation** — if you have an iPhone, run Live Link Face on it for ARKit-grade blendshapes; it rides the same EVMC4U skeleton or a parallel face-only pipe.
- **NDI alternatives** — if NDI is finicky, you can use **Window Capture** of the Unreal viewport in OBS (slightly higher latency, simpler).
- **Running on Linux** — Unreal Engine 5 on Linux works but the install is rougher. The Python pipeline is fully cross-platform.
