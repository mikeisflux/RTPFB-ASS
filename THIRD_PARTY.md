# Third-Party Components & Licenses

This project integrates with several upstream repos and pre-trained models. We
do **not vendor** their source into this repository — `scripts/install_third_party.sh`
clones them into `third_party/` (git-ignored) so each stays under its own license.

If you import / link / redistribute any of the components below, the obligations
of *their* license apply to your build.

## Cloned upstream repos

| Repo | License | Notes |
|---|---|---|
| [iperov/DeepFaceLive](https://github.com/iperov/DeepFaceLive) | GPL-3.0 | Strong copyleft. Code that imports DeepFaceLive becomes GPL-3.0. We integrate via subprocess only. |
| [facefusion/facefusion](https://github.com/facefusion/facefusion) | OSL-3.0 + non-commercial restrictions | Read their LICENSE before commercial use. |
| [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Research / non-commercial | Authors restrict to research use. |
| [HumanAIGC/AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone) | Research code (varies by fork) | Many forks; check the specific fork you clone. |
| [magic-research/magicanimate](https://github.com/magic-research/magicanimate) | BSD-3-Clause (code) — model weights have separate terms | |
| [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | INRIA research license — non-commercial | Optional; not pulled by default. |

## Pre-trained model weights

| Model | Source | License |
|---|---|---|
| `inswapper_128.onnx` | InsightFace / Picsi.AI | Non-commercial use only. |
| InsightFace `buffalo_l` detection / recognition pack | InsightFace | Apache-2.0 (code) — model weights non-commercial. |
| `wav2lip_gan.pth` | Rudrabha/Wav2Lip releases | Research / non-commercial. |
| MediaPipe Holistic | Google | Apache-2.0. Permissive. |

## What this means in practice

- **Personal / streaming / research use:** generally fine across the stack.
- **Commercial product:** you must replace the non-commercial models with ones
  you have rights to (your own trained model, a licensed FaceFusion build, or a
  cleanroom equivalent), and you must avoid linking to GPL-3.0 code from
  proprietary modules.
- **This repo's own code** (everything under `src/rtpfb/` and `scripts/`) is yours
  to license as you choose — pick a `LICENSE` file before publishing.
