param(
    [int]$Port = 2024,
    [string]$AdbPath = "",
    [switch]$EnsureFirewallRule
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title =="
}

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Convert-LocalPropertiesPath {
    param([string]$Value)
    $path = $Value.Trim()
    $path = $path -replace "\\:", ":"
    $path = $path -replace "\\\\", "\"
    return $path
}

function Find-Adb {
    if ($AdbPath) {
        if (Test-Path -LiteralPath $AdbPath) { return (Resolve-Path -LiteralPath $AdbPath).Path }
        throw "ADB path does not exist: $AdbPath"
    }

    $fromPath = Get-Command adb.exe -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }

    $candidates = @()
    if ($env:ANDROID_HOME) { $candidates += (Join-Path $env:ANDROID_HOME "platform-tools\adb.exe") }
    if ($env:ANDROID_SDK_ROOT) { $candidates += (Join-Path $env:ANDROID_SDK_ROOT "platform-tools\adb.exe") }
    if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe") }

    $frontendLocalProperties = Join-Path (Split-Path (Get-RepoRoot) -Parent) "AIGC_Figma_Frontend\local.properties"
    if (Test-Path -LiteralPath $frontendLocalProperties) {
        $sdkLine = Get-Content -LiteralPath $frontendLocalProperties |
            Where-Object { $_ -match "^sdk\.dir=" } |
            Select-Object -First 1
        if ($sdkLine) {
            $sdkDir = Convert-LocalPropertiesPath ($sdkLine -replace "^sdk\.dir=", "")
            $candidates += (Join-Path $sdkDir "platform-tools\adb.exe")
        }
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return $null
}

function Get-LanAddresses {
    $ignoredInterfacePattern = "Loopback|vEthernet|VMware|VirtualBox|Hyper-V|Meta|Tailscale|ZeroTier"
    Get-NetIPConfiguration |
        Where-Object {
            $_.NetAdapter.Status -eq "Up" -and
            $_.IPv4DefaultGateway -and
            $_.IPv4Address -and
            $_.InterfaceAlias -notmatch $ignoredInterfacePattern
        } |
        ForEach-Object {
            foreach ($address in $_.IPv4Address) {
                if (
                    $address.IPAddress -and
                    $address.IPAddress -notlike "169.254.*" -and
                    $address.IPAddress -notlike "198.18.*" -and
                    $address.IPAddress -notlike "198.19.*"
                ) {
                    [PSCustomObject]@{
                        InterfaceAlias = $_.InterfaceAlias
                        Address = $address.IPAddress
                    }
                }
            }
        }
}

Write-Section "Server listener"
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listeners) {
    Write-Warning "Nothing is listening on port $Port. Start the server with: langgraph dev --host 0.0.0.0 --port $Port"
} else {
    $listeners | Select-Object LocalAddress, LocalPort, State, OwningProcess | Format-Table -AutoSize
}

Write-Section "LAN URLs"
$lanAddresses = @(Get-LanAddresses)
if (-not $lanAddresses) {
    Write-Warning "No active IPv4 LAN address with a gateway was found."
} else {
    $lanAddresses | ForEach-Object {
        Write-Host ("Use from same LAN ({0}): http://{1}:{2}" -f $_.InterfaceAlias, $_.Address, $Port)
    }
}
Write-Host ("Use after adb reverse: http://127.0.0.1:{0}" -f $Port)

Write-Section "Local HTTP check"
$httpCheckHosts = @("127.0.0.1") + @($lanAddresses | ForEach-Object { $_.Address })
foreach ($hostName in $httpCheckHosts) {
    try {
        $url = "http://${hostName}:$Port/adb/status"
        $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5
        Write-Host ("OK {0} -> HTTP {1} {2}" -f $url, $response.StatusCode, $response.Content)
    } catch {
        Write-Warning ("FAILED http://${hostName}:$Port/adb/status -> {0}" -f $_.Exception.Message)
    }
}

Write-Section "Firewall rule"
$ruleName = "LangGraph $Port"
$ruleText = (& netsh advfirewall firewall show rule name="$ruleName" verbose) 2>$null
if ($LASTEXITCODE -eq 0 -and ($ruleText -match "LocalPort:\s+$Port") -and ($ruleText -match "Action:\s+Allow")) {
    Write-Host "Firewall allow rule exists: $ruleName"
} elseif ($EnsureFirewallRule) {
    & netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=TCP localport=$Port profile=any
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to add firewall rule. Re-run PowerShell as Administrator."
    }
    Write-Host "Added firewall allow rule: $ruleName"
} else {
    Write-Warning "No confirmed firewall allow rule for TCP $Port."
    Write-Host "To add it, run this script from an Administrator PowerShell with -EnsureFirewallRule."
}

Write-Section "ADB reverse"
$adb = Find-Adb
if (-not $adb) {
    Write-Warning "adb.exe was not found. Install Android SDK Platform-Tools or pass -AdbPath."
    exit 0
}

Write-Host "ADB: $adb"
$devicesOutput = & $adb devices
$devicesOutput | ForEach-Object { Write-Host $_ }
$onlineDevices = @(
    $devicesOutput |
        Where-Object { $_ -match "^\S+\s+device$" } |
        ForEach-Object { ($_ -split "\s+")[0] }
)

if (-not $onlineDevices) {
    Write-Warning "No authorized USB device found. Connect the phone, enable USB debugging, and accept the RSA prompt."
    exit 0
}

& $adb reverse "tcp:$Port" "tcp:$Port"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to configure adb reverse."
}
Write-Host "ADB reverse configured: phone tcp:$Port -> computer tcp:$Port"
& $adb reverse --list
Write-Host ""
Write-Host ("In the Android app, set service URL to: http://127.0.0.1:{0}" -f $Port)
