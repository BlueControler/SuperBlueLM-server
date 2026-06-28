param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 2024
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$langgraph = Join-Path $repoRoot ".venv\Scripts\langgraph.exe"

if (-not (Test-Path -LiteralPath $langgraph)) {
    throw "langgraph.exe was not found at $langgraph. Create or repair the project virtual environment first."
}

# Force UTF-8 so python-dotenv can read .env files that contain Chinese comments
# when Windows PowerShell would otherwise default to the system ANSI code page.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host ("Starting LangGraph on http://{0}:{1}" -f $HostAddress, $Port)
Write-Host "For Wi-Fi direct mode, use the computer's WLAN IPv4 address in the Android app."

& $langgraph dev --host $HostAddress --port $Port
