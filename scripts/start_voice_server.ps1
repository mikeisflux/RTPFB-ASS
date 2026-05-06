# Bootstrap (idempotent) and start the w-okada voice-changer server on
# Windows / PowerShell. First run sets up an isolated venv just for w-okada
# and installs its pinned deps. Subsequent runs just start the server.
#
# Usage:  .\scripts\start_voice_server.ps1
# Options:
#   -Cpu       Skip CUDA wheels, install CPU-only torch (slow but works without an NVIDIA GPU)
#   -Port N    Port to bind (default 18888)
#   -Reinstall Force-reinstall deps even if the venv already exists

param(
    [switch]$Cpu,
    [int]$Port = 18888,
    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"

$RepoRoot      = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ServerDir     = Join-Path $RepoRoot "third_party\voice-changer\server"
$VenvDir       = Join-Path $ServerDir ".venv-wokada"
$VenvPython    = Join-Path $VenvDir "Scripts\python.exe"
$Requirements  = Join-Path $ServerDir "requirements.txt"
$ServerEntry   = Join-Path $ServerDir "MMVCServerSIO.py"

if (-not (Test-Path $ServerEntry)) {
    throw "voice-changer server not found at $ServerDir. Run scripts\install_third_party.ps1 first."
}

# --- venv ------------------------------------------------------------------
if ($Reinstall -and (Test-Path $VenvDir)) {
    Write-Host "[setup] Reinstall requested, removing existing venv..."
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[setup] Creating venv at $VenvDir"
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    $NeedsInstall = $true
} else {
    # Check whether key deps are importable; if not, treat as fresh install.
    & $VenvPython -c "import torch, fastapi, faiss, librosa, onnxruntime" 2>$null
    $NeedsInstall = ($LASTEXITCODE -ne 0)
}

# --- deps ------------------------------------------------------------------
if ($NeedsInstall) {
    Write-Host "[setup] Upgrading pip..."
    & $VenvPython -m pip install --upgrade pip --quiet

    if ($Cpu) {
        Write-Host "[setup] Installing torch 2.0.1 (CPU)..."
        & $VenvPython -m pip install torch==2.0.1 torchaudio==2.0.2
    } else {
        Write-Host "[setup] Installing torch 2.0.1 + torchaudio 2.0.2 (CUDA 11.8 wheels)..."
        & $VenvPython -m pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[setup] CUDA wheels failed, falling back to CPU wheels..."
            & $VenvPython -m pip install torch==2.0.1 torchaudio==2.0.2
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

    Write-Host "[setup] Installing the rest of requirements.txt..."
    & $VenvPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] requirements install failed - this is usually onnxruntime-gpu pinning to an old CUDA."
        Write-Host "[setup] Retrying with onnxruntime (CPU) instead of onnxruntime-gpu..."
        & $VenvPython -m pip install onnxruntime==1.13.1
        & $VenvPython -m pip install -r $Requirements --no-deps
        if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
    }
}

# --- sanity ----------------------------------------------------------------
& $VenvPython -c @"
import torch, fastapi, faiss, librosa
try:
    import onnxruntime
    rt = 'onnxruntime ' + onnxruntime.__version__
except ImportError:
    rt = 'onnxruntime missing'
print(f'[ok] torch={torch.__version__} cuda_available={torch.cuda.is_available()} {rt}')
"@
if ($LASTEXITCODE -ne 0) { throw "dep sanity check failed - try -Reinstall" }

# --- run -------------------------------------------------------------------
Write-Host ""
Write-Host "================================================================"
Write-Host "  Starting w-okada voice-changer server"
Write-Host "  URL:    http://localhost:$Port"
Write-Host "  Stop:   Ctrl+C in this window"
Write-Host ""
Write-Host "  Once the UI loads, point a slot at:"
Write-Host "    voices\egirl\rvc\model.pth"
Write-Host "    voices\egirl\rvc\added.index"
Write-Host "================================================================"
Write-Host ""

Push-Location $ServerDir
try {
    & $VenvPython MMVCServerSIO.py -p $Port --https false
}
finally {
    Pop-Location
}
