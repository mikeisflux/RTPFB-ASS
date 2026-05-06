# CLAUDE.md

## Environment

- **Operating system:** Windows
- **Shell:** PowerShell (NOT bash, NOT cmd). The user runs commands from a `.venv`-activated PowerShell prompt rooted at `C:\Users\Mike\RTPFB-ASS\RTPFB-ASS\`.
- All shell snippets you give the user MUST be valid PowerShell. Do not assume bash, zsh, or POSIX tools are available.

## PowerShell rules to follow

- Use backticks `` ` `` for line continuation, not backslashes `\`.
- Use `\` (not `/`) in Windows file paths inside quoted strings.
- `Get-ChildItem -Filter` accepts ONE pattern only. For multiple extensions, use `Where-Object { $_.Extension -in '.txt','.py' }` or pipe through `Where-Object { $_.Name -like '*.txt' -or $_.Name -like '*.py' }`. Do NOT pass a comma-separated list to `-Filter`.
- `Test-Path <path>` for existence checks (not `[ -e ]` or `test -e`).
- Activate venvs with `.\.venv\Scripts\Activate.ps1` (not `source .venv/bin/activate`).
- Run `.ps1` scripts as `powershell -ExecutionPolicy Bypass -File scripts\name.ps1` if execution policy blocks them.
- For "does this file exist" prefer `Test-Path`; for "list matching files" prefer `Get-ChildItem` with `-Filter` (single pattern) or piped `Where-Object`.
- Quote paths with spaces using double quotes: `"C:\Program Files\..."`.
- Environment variables: `$env:VAR_NAME` (not `$VAR_NAME` or `%VAR_NAME%`).

## Repo layout reminders

- `voices/` — voice library, written by `rtpfb voice import` / `rtpfb voice add`. Each voice lives in `voices/<name>/{rvc/,knnvc/,meta.json}`.
- `third_party/` — gitignored. Cloned by `scripts\install_third_party.ps1`. Holds w-okada/voice-changer, RVC-WebUI, knn-vc, etc.
- `models/`, `assets/` — gitignored asset dirs computed off `REPO_ROOT` in `src/rtpfb/config.py`.
- `rtpfb` CLI is installed into the project `.venv` and figures out repo root from its install location, so commands work regardless of CWD.

## Voice pipeline

- In-process RVC inference is **stubbed**. To use an imported RVC voice live, run the **w-okada/voice-changer** server in a separate PowerShell window (with its own venv — its deps fight `rtpfb`'s) and pass `--voice-backend w-okada --voice-ws ws://localhost:18888` to `rtpfb run`.
- KNN-VC works in-process today via `rtpfb voice add ... --method knn-vc`.

## Working branch

Develop on `claude/merge-setup-fix-upload-G5TKm`. Push there, never to `main` without explicit permission.
