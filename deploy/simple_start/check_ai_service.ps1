Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "Check Defluor AI service"
Write-Host "=========================================="
Write-Host ""

$health = $null
$ready = $null

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 3
} catch {
    $health = $null
}

try {
    $ready = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/ready" -TimeoutSec 3
} catch {
    $ready = $null
}

if ($health) {
    Write-Host "[OK] API service is running"
} else {
    Write-Host "[FAIL] API service is not reachable"
    Write-Host "Reason: the start window may not be open, or the API failed during startup."
    Write-Host "Action: double-click the root start script and keep that window open."
    Write-Host ""
    exit 1
}

if ($ready) {
    if ($ready.status -eq "ready") {
        Write-Host "[READY]"
        $ready | ConvertTo-Json -Depth 5
    } else {
        Write-Host "[NOT READY]"
        $ready | ConvertTo-Json -Depth 5
        Write-Host "Action: wait a little longer, then run this check again."
    }
} else {
    Write-Host "[FAIL] model is not ready"
    Write-Host "Action: wait a little longer, then run this check again."
}

Write-Host ""
Write-Host "If FAIL is shown, double-click the start script first."
