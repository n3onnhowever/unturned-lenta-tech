$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

docker compose ps
Write-Host "\nBackend health:" -ForegroundColor Cyan
curl.exe http://localhost:8000/health
Write-Host "\nML health:" -ForegroundColor Cyan
curl.exe http://localhost:8000/health/ml
