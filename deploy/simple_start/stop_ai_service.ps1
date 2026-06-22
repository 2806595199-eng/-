Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

Write-Host "=========================================="
Write-Host "Stop Defluor AI service"
Write-Host "=========================================="
Write-Host ""
Write-Host "This stops the Python/uvicorn process listening on port 8000."
Write-Host "Do not use this script if another service is using port 8000."
Write-Host ""

$conns = Get-NetTCPConnection -LocalPort 8000 -State Listen
if (-not $conns) {
    Write-Host "No service is listening on port 8000."
    exit 0
}

$owners = $conns | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($owner in $owners) {
    $proc = Get-Process -Id $owner
    if ($proc) {
        Write-Host ("Stopping process: " + $proc.ProcessName + " PID=" + $owner)
        Stop-Process -Id $owner -Force
    }
}

Write-Host ""
Write-Host "Stop command finished."
