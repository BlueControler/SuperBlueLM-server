param(
    [int]$Port = 2024,
    [int]$Seconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-IPv4AddressesFromIpconfig {
    $addresses = @()
    $lines = & ipconfig.exe
    foreach ($line in $lines) {
        if ($line -match "IPv4.*?:\s*([0-9]{1,3}(\.[0-9]{1,3}){3})") {
            $ip = $matches[1]
            if ($ip -notlike "127.*" -and $ip -notlike "169.254.*" -and $ip -notlike "198.18.*" -and $ip -notlike "198.19.*") {
                $addresses += $ip
            }
        }
    }
    return @($addresses | Select-Object -Unique)
}

function Test-LocalEndpoint {
    param([string]$HostName)
    $url = "http://${HostName}:$Port/adb/status"
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5
        Write-Host ("OK {0} -> HTTP {1} {2}" -f $url, $response.StatusCode, $response.Content)
    } catch {
        Write-Warning ("FAILED {0} -> {1}" -f $url, $_.Exception.Message)
    }
}

function Get-PortConnections {
    $rows = & netstat.exe -ano -p TCP
    foreach ($row in $rows) {
        $trimmed = $row.Trim()
        if ($trimmed -notmatch "^TCP\s+") { continue }

        $parts = $trimmed -split "\s+"
        if ($parts.Count -lt 5) { continue }

        $local = $parts[1]
        $remote = $parts[2]
        $state = $parts[3]
        $pidValue = $parts[4]

        if ($local -notmatch ":$Port$") { continue }
        if ($state -eq "LISTENING") { continue }
        if ($remote -match "^(0\.0\.0\.0|\*):0$") { continue }

        [PSCustomObject]@{
            Local = $local
            Remote = $remote
            State = $state
            PID = $pidValue
        }
    }
}

$addresses = Get-IPv4AddressesFromIpconfig
Write-Host "Wi-Fi/LAN direct access monitor"
Write-Host ("Port: {0}" -f $Port)
Write-Host ""

if (-not $addresses) {
    Write-Warning "No LAN IPv4 address found from ipconfig."
} else {
    Write-Host "Try these URLs from the phone browser while this script is running:"
    foreach ($address in $addresses) {
        Write-Host ("  http://{0}:{1}/adb/status" -f $address, $Port)
    }
}

Write-Host ""
Write-Host "Local endpoint self-check:"
Test-LocalEndpoint "127.0.0.1"
foreach ($address in $addresses) {
    Test-LocalEndpoint $address
}

Write-Host ""
Write-Host ("Monitoring incoming TCP connections for {0} seconds..." -f $Seconds)
Write-Host "If the phone browser is loading but no remote address appears here, the Wi-Fi network is blocking peer-to-peer access."
Write-Host ""

$deadline = (Get-Date).AddSeconds($Seconds)
$seen = @{}
$hadHit = $false

while ((Get-Date) -lt $deadline) {
    foreach ($connection in Get-PortConnections) {
        $key = "{0}|{1}|{2}" -f $connection.Remote, $connection.State, $connection.PID
        if (-not $seen.ContainsKey($key)) {
            $seen[$key] = $true
            $hadHit = $true
            Write-Host ("{0} remote={1} state={2} pid={3}" -f (Get-Date -Format "HH:mm:ss"), $connection.Remote, $connection.State, $connection.PID)
        }
    }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
if ($hadHit) {
    Write-Host "Observed at least one incoming TCP connection on this port."
    Write-Host "If the Android app still fails, compare the app URL with the browser URL and check app-side logs."
} else {
    Write-Warning "No incoming TCP connection was observed."
    Write-Host "For Wi-Fi mode, connect the phone and PC to a non-isolated hotspot/router, then use the shown LAN URL in the Android app."
}
