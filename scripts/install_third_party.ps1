# PowerShell equivalent of install_third_party.sh — for Windows users without
# WSL or Git Bash on PATH. Clones every upstream repo we integrate with into
# ./third_party/ (gitignored). See THIRD_PARTY.md for license notes.
#
# Usage:  pwsh scripts/install_third_party.ps1
# Or in PowerShell: powershell -ExecutionPolicy Bypass -File scripts\install_third_party.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dest = Join-Path $RepoRoot "third_party"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Clone-IfMissing {
    param(
        [Parameter(Mandatory)] [string]$Url,
        [Parameter(Mandatory)] [string]$Dir
    )
    $target = Join-Path $Dest $Dir
    if (Test-Path (Join-Path $target ".git")) {
        Write-Host "[skip]  $Dir already cloned"
        return
    }
    Write-Host "[clone] $Url -> $Dir"
    & git clone --depth 1 $Url $target
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed for $Url"
    }
}

Clone-IfMissing "https://github.com/iperov/DeepFaceLive.git"                                "DeepFaceLive"
Clone-IfMissing "https://github.com/facefusion/facefusion.git"                              "facefusion"
Clone-IfMissing "https://github.com/Rudrabha/Wav2Lip.git"                                   "Wav2Lip"
Clone-IfMissing "https://github.com/OpenTalker/SadTalker.git"                               "SadTalker"
Clone-IfMissing "https://github.com/HumanAIGC/AnimateAnyone.git"                            "AnimateAnyone"
Clone-IfMissing "https://github.com/magic-research/magicanimate.git"                        "magicanimate"
Clone-IfMissing "https://github.com/KwaiVGI/LivePortrait.git"                               "LivePortrait"
Clone-IfMissing "https://github.com/w-okada/voice-changer.git"                              "voice-changer"
Clone-IfMissing "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git" "RVC-WebUI"
Clone-IfMissing "https://github.com/yxlllc/DDSP-SVC.git"                                    "DDSP-SVC"
Clone-IfMissing "https://github.com/bshall/knn-vc.git"                                      "knn-vc"

Write-Host ""
Write-Host "Done. Upstream repos are in $Dest"
Write-Host "Each retains its own license — see THIRD_PARTY.md."
