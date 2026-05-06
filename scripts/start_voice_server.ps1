# Bootstrap (idempotent) and start the w-okada voice-changer server on
# Windows / PowerShell. First run sets up an isolated venv just for w-okada
# and installs its pinned deps. Subsequent runs just start the server.
#
# Usage:  .\scripts\start_voice_server.ps1
# Options:
#   -Cpu        Skip CUDA wheels, install CPU-only torch (works without NVIDIA GPU but slow)
#   -Nightly    Install torch nightly with CUDA 12.8 wheels. Required for Blackwell GPUs
#               (RTX 50-series / sm_120) - torch 2.0.1's pinned cu118 wheels don't support sm_120.
#               Also upgrades onnxruntime-gpu to a Blackwell-compatible version.
#   -Port N     Port to bind (default 18888)
#   -Reinstall  Force-reinstall deps even if the venv already exists

param(
    [switch]$Cpu,
    [switch]$Nightly,
    [int]$Port = 18888,
    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"

# Run a Python import probe without letting a failed import (which writes a
# traceback to stderr) terminate the script. With $ErrorActionPreference=Stop,
# native commands writing to stderr raise NativeCommandError even when
# stderr is redirected. This helper sidesteps that.
function Test-PyImport {
    param([string]$PythonExe, [string]$ModuleList)
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $PythonExe -c "import $ModuleList" 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $saved
    }
}

$RepoRoot      = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ServerDir     = Join-Path $RepoRoot "third_party\voice-changer\server"
$VenvDir       = Join-Path $ServerDir ".venv-wokada"
$VenvPython    = Join-Path $VenvDir "Scripts\python.exe"
$Requirements  = Join-Path $ServerDir "requirements.txt"
$ServerEntry   = Join-Path $ServerDir "MMVCServerSIO.py"

if (-not (Test-Path $ServerEntry)) {
    throw "voice-changer server not found at $ServerDir. Run scripts\install_third_party.ps1 first."
}
if ($Cpu -and $Nightly) {
    throw "-Cpu and -Nightly are mutually exclusive."
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
    $NeedsInstall = -not (Test-PyImport $VenvPython "torch, fastapi, faiss, librosa, onnxruntime")
}

# In -Nightly mode, also force a torch upgrade if the installed torch is the
# old cu118 build that doesn't support Blackwell.
$NeedsTorchUpgrade = $false
if (-not $NeedsInstall -and $Nightly) {
    $TorchInfo = & $VenvPython -c "import torch; print(torch.__version__)" 2>$null
    if ($TorchInfo -like "2.0.1*" -or $TorchInfo -like "*+cu118*") {
        Write-Host "[setup] -Nightly requested but venv has $TorchInfo - upgrading torch."
        $NeedsTorchUpgrade = $true
    }
}

# --- deps ------------------------------------------------------------------
if ($NeedsInstall) {
    Write-Host "[setup] Upgrading pip..."
    & $VenvPython -m pip install --upgrade pip --quiet

    if ($Nightly) {
        # Install everything from requirements.txt EXCEPT torch / torchaudio /
        # onnxruntime-gpu (we want newer versions of those for Blackwell).
        $FilteredReqs = Join-Path $env:TEMP "wokada_requirements_no_torch.txt"
        Get-Content $Requirements | Where-Object {
            $_ -notmatch '^\s*(torch|torchaudio|onnxruntime-gpu)\s*=='
        } | Out-File $FilteredReqs -Encoding utf8

        Write-Host "[setup] Installing non-torch requirements..."
        & $VenvPython -m pip install -r $FilteredReqs
        if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

        Write-Host "[setup] Installing torch nightly (CUDA 12.8, sm_120 / Blackwell support)..."
        & $VenvPython -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
        if ($LASTEXITCODE -ne 0) { throw "torch nightly install failed" }

        Write-Host "[setup] Installing latest onnxruntime-gpu (CUDA 12.x, Blackwell-compatible)..."
        & $VenvPython -m pip install --upgrade onnxruntime-gpu
        if ($LASTEXITCODE -ne 0) { throw "onnxruntime-gpu upgrade failed" }
    }
    elseif ($Cpu) {
        Write-Host "[setup] Installing torch 2.0.1 (CPU)..."
        & $VenvPython -m pip install torch==2.0.1 torchaudio==2.0.2
        if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
        Write-Host "[setup] Installing the rest of requirements.txt..."
        & $VenvPython -m pip install -r $Requirements
        if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
    }
    else {
        Write-Host "[setup] Installing torch 2.0.1 + torchaudio 2.0.2 (CUDA 11.8 wheels)..."
        Write-Host "[setup] NOTE: if you have an RTX 50-series (Blackwell, sm_120) GPU, this will not"
        Write-Host "[setup]       use the GPU. Re-run with -Nightly -Reinstall to fix."
        & $VenvPython -m pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[setup] CUDA wheels failed, falling back to CPU wheels..."
            & $VenvPython -m pip install torch==2.0.1 torchaudio==2.0.2
            if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
        }
        Write-Host "[setup] Installing the rest of requirements.txt..."
        & $VenvPython -m pip install -r $Requirements
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[setup] requirements install failed - usually onnxruntime-gpu pinning to old CUDA."
            Write-Host "[setup] Retrying with onnxruntime (CPU) instead of onnxruntime-gpu..."
            & $VenvPython -m pip install onnxruntime==1.13.1
            & $VenvPython -m pip install -r $Requirements --no-deps
            if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
        }
    }
}
elseif ($NeedsTorchUpgrade) {
    Write-Host "[setup] Upgrading torch to nightly cu128 in existing venv..."
    & $VenvPython -m pip uninstall -y torch torchaudio
    & $VenvPython -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    if ($LASTEXITCODE -ne 0) { throw "torch nightly install failed" }
    Write-Host "[setup] Upgrading onnxruntime-gpu to latest (Blackwell-compatible)..."
    & $VenvPython -m pip install --upgrade onnxruntime-gpu
    if ($LASTEXITCODE -ne 0) { throw "onnxruntime-gpu upgrade failed" }
}

# --- fairseq (idempotent) --------------------------------------------------
# w-okada's RVC code imports fairseq.checkpoint_utils to load the hubert /
# contentvec audio embedder, but fairseq is NOT in their requirements.txt.
# Install with --no-deps because fairseq pins torch<2 and we'd otherwise lose
# our nightly cu128 install. Then install fairseq's actual runtime Python
# deps separately (minus torch/torchaudio which we manage).
if (-not (Test-PyImport $VenvPython "fairseq")) {
    Write-Host "[setup] Installing fairseq (missing from w-okada's requirements.txt)..."
    & $VenvPython -m pip install "fairseq==0.12.2" --no-deps
    if ($LASTEXITCODE -ne 0) { throw "fairseq install failed" }
}

# fairseq's runtime deps - install separately so we can use modern versions.
# Don't pin hydra-core<1.1: that forces omegaconf<2.1, whose wheels have
# invalid PEP 440 metadata (PyYAML (>=5.1.*)) that pip>=24.1 rejects.
# fairseq.checkpoint_utils works fine against modern hydra/omegaconf for
# what w-okada actually does (loading a hubert checkpoint).
if (-not (Test-PyImport $VenvPython "omegaconf, hydra, bitarray")) {
    Write-Host "[setup] Installing fairseq runtime deps (modern hydra/omegaconf)..."
    & $VenvPython -m pip install bitarray omegaconf hydra-core sacrebleu portalocker regex cffi cython
    if ($LASTEXITCODE -ne 0) { throw "fairseq runtime deps install failed" }
}

# pyworld is used by w-okada's Dio/Harvest pitch extractors but isn't in
# their requirements.txt either. Pre-built Windows wheels exist on PyPI.
if (-not (Test-PyImport $VenvPython "pyworld")) {
    Write-Host "[setup] Installing pyworld (missing from w-okada's requirements.txt)..."
    & $VenvPython -m pip install pyworld
    if ($LASTEXITCODE -ne 0) { throw "pyworld install failed" }
}

# --- sanity ----------------------------------------------------------------
& $VenvPython -c @"
import torch, fastapi, faiss, librosa, fairseq, pyworld
try:
    import onnxruntime
    rt = f'onnxruntime {onnxruntime.__version__}'
except ImportError:
    rt = 'onnxruntime missing'
cap = ''
if torch.cuda.is_available():
    try:
        major, minor = torch.cuda.get_device_capability(0)
        cap = f' sm_{major}{minor}'
    except Exception:
        pass
print(f'[ok] torch={torch.__version__} cuda_available={torch.cuda.is_available()}{cap} {rt} fairseq={fairseq.__version__}')
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
Write-Host ""
Write-Host "  Note: '[Voice Changer] Client Launch Exception, [WinError 2]'"
Write-Host "  is harmless - it's w-okada looking for its Electron desktop"
Write-Host "  client which we don't ship. Use the browser UI."
Write-Host "================================================================"
Write-Host ""

Push-Location $ServerDir
try {
    & $VenvPython MMVCServerSIO.py -p $Port --https false
}
finally {
    Pop-Location
}
