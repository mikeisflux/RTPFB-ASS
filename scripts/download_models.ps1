# PowerShell equivalent of download_models.sh — for Windows users without
# WSL or Git Bash on PATH.
#
# Usage:  pwsh scripts/download_models.ps1
# Or in PowerShell: powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelsDir = Join-Path $RepoRoot "models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

@"
Place the following files in: $ModelsDir

  inswapper_128.onnx       — face swap (faceswap mode)
                             Source: InsightFace / Picsi.AI release
                             License: NON-COMMERCIAL

  buffalo_l/               — InsightFace detection + recognition pack
                             Auto-downloaded by insightface on first run.

  wav2lip_gan.pth          — lip sync (faceswap mode, optional)
                             Source: Rudrabha/Wav2Lip releases
                             License: research / non-commercial

  face_landmarker.task     — MediaPipe Face Landmarker v2 (mocap mode)
                             52 ARKit-standard blendshapes natively.
                             License: Apache-2.0. Auto-fetched below.

By placing these files in the models/ directory you affirm you have rights
to their use under each model's license.
"@ | Write-Host

# Auto-fetch the Apache-licensed Face Landmarker model since it's needed
# for the recommended mocap path.
$FaceTask = Join-Path $ModelsDir "face_landmarker.task"
if (-not (Test-Path $FaceTask)) {
    Write-Host ""
    Write-Host "Fetching face_landmarker.task (MediaPipe Face Landmarker v2, Apache-2.0)…"
    try {
        $url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        Invoke-WebRequest -Uri $url -OutFile $FaceTask -UseBasicParsing
        Write-Host "  saved → $FaceTask"
    } catch {
        Write-Host "  (skipped — network unavailable; download manually later)"
        Write-Host "  $($_.Exception.Message)"
    }
} else {
    Write-Host ""
    Write-Host "[skip] face_landmarker.task already present at $FaceTask"
}
