# Full-Body Photoreal Avatar via Unreal Engine 5 + MetaHuman

This is the **mocap mode** stack: our Python pipeline does the motion capture
and voice conversion, Unreal does the rendering and physics. Everything in
this guide is free.

## Why this stack

| Requirement | How it's satisfied |
|---|---|
| Photorealistic female body | MetaHuman Creator (free) |
| Real-time, single GPU | Unreal 5 Lumen + Nanite, easily real-time on a 5090 |
| Soft-body chest squish on lean | Chaos Cloth on a chest cluster, or Chaos Flesh (UE 5.4+) |
| Cloth physics | Chaos Cloth (built into UE5) |
| Mocap from a webcam | Our Python pipeline → VMC over UDP → EVMC4U plugin |
| Output to OBS / Twitch | NDI plugin or Window Capture of the Unreal viewport |
| Voice cloning | Our Python pipeline (RVC / KNN-VC) → virtual mic → OBS |
| Cost | $0 |

## Stack diagram

```
webcam ─► python (rtpfb run --mode mocap)
              │
              ├─► MediaPipe Holistic (body, face, hands)
              │       │
              │       └─► VMC over UDP :39539 ─►  Unreal 5
              │                                     │
              │                                     ├─► EVMC4U plugin
              │                                     │   maps VMC bones →
              │                                     │   MetaHuman skeleton
              │                                     │
              │                                     ├─► Chaos Cloth on
              │                                     │   chest / cloth bones
              │                                     │
              │                                     └─► NDI or Window
              │                                         Capture ─► OBS
              │
              └─► RVC voice changer ─► virtual mic ─► OBS

                                                       OBS ─► Twitch / YouTube
```

## One-time setup

### 1. Unreal Engine 5

1. Install **Epic Games Launcher** → install **Unreal Engine 5.4+** (free).
2. Create a new project: `Games > Blank > Blueprint, with Starter Content disabled, Maximum Quality`.

### 2. MetaHuman

1. In Unreal: `Window > Quixel Bridge` → log in with the free Epic account.
2. `MetaHumans` tab → either pick a stock female MetaHuman or click **MetaHuman Creator** to design one. All free.
3. Drag the MetaHuman into your level.

### 3. EVMC4U (free VMC receiver plugin for Unreal)

1. Clone https://github.com/HAL9HARUKU/EVMC4U into `<YourProject>/Plugins/EVMC4U`.
2. Restart Unreal; Plugins panel should show **EVMC4U enabled**.
3. In your project settings, enable the OSC plugin (it's a dependency).

### 4. Wire EVMC4U to the MetaHuman skeleton

1. Make a new Blueprint based on `Actor`. Add an **EVMC4U Manager** component.
2. Set its port to `39539` (matches our default).
3. In the manager, set the target skeletal mesh to your MetaHuman's body mesh.
4. EVMC4U has a built-in bone-name remap table — set it to the **VRM → MetaHuman** preset (community presets exist on the EVMC4U repo wiki) or map by hand: VMC's `Hips` → MetaHuman's `pelvis`, `LeftUpperArm` → `upperarm_l`, etc.

### 5. Chaos Cloth on the chest

This is what gives you the squish-on-lean.

1. Open the MetaHuman body mesh asset.
2. Switch to the **Cloth** asset editor.
3. Mask the chest region as cloth-driven; keep the rest kinematic.
4. Tune `BendingStiffness`, `Damping`, and `BackstopRadius` until the lean response looks right.
5. (Optional, UE 5.4+) use **Chaos Flesh** for actual volumetric soft body — heavier but more realistic compression.

### 6. Output to OBS

Pick one:

* **NDI** (recommended). Install [obs-ndi](https://github.com/obs-ndi/obs-ndi) in OBS and the [NDI plugin](https://github.com/ufna/UnrealNDI) in Unreal. Frame-accurate, sub-frame latency.
* **Window Capture**. Capture the Unreal Editor's viewport / a packaged build window. Simpler, slightly higher latency.

## Run the streaming session

```bash
# 1. Start Unreal, hit Play in the level (EVMC4U starts listening).

# 2. In a terminal:
rtpfb run --preset presets/mocap-unreal.yaml --voice-model alice
```

The Python side ships pose data + your converted voice. Unreal does the
rendering and physics. OBS picks up Unreal's NDI feed for video and the
loopback device for audio.

## Tuning notes

* Out-of-the-box VMC sends position-only bones with identity rotations; the
  renderer's IK solver fills in joint angles. If hand orientation is off,
  enable rotation derivation (TODO) or rely on Unreal Control Rig IK.
* MediaPipe's z-coordinate is noisy. For best results, lock the avatar to
  a fixed-radius cylinder around the hips and let the IK absorb depth jitter.
* Face blendshapes from MediaPipe are approximate. If lip sync looks
  wrong, switch to Unreal's **Live Link Face** (iPhone ARKit) for the face
  channel and keep our pipeline only for body + voice.

## Faceswap fallback

If you can't run Unreal, the project still has the legacy 2D path:

```bash
rtpfb run --mode faceswap --target ./assets/face.png --voice-model alice
```

That gives you a face swap into a virtual camera — no body, no physics.
