param(
  [string]$VideoPath
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Checking backend..." -ForegroundColor Cyan
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/health/ml

if (-not $VideoPath) {
  Write-Host "\nBackend is healthy. To run API upload smoke:" -ForegroundColor Yellow
  Write-Host "powershell -ExecutionPolicy Bypass -File scripts/run-smoke-e2e.ps1 -VideoPath C:\path\to\small_video.mp4"
  Write-Host "\nThen open frontend in backend mode and repeat the same upload through UI."
  exit 0
}

if (-not (Test-Path $VideoPath)) {
  throw "Video not found: $VideoPath"
}

Write-Host "Uploading $VideoPath" -ForegroundColor Cyan
$response = curl.exe -s -F "file=@$VideoPath" http://localhost:8000/jobs/upload | ConvertFrom-Json
$jobId = $response.job_id
Write-Host "job_id: $jobId"

$status = ""
for ($i = 0; $i -lt 120; $i++) {
  Start-Sleep -Seconds 3
  $job = curl.exe -s "http://localhost:8000/jobs/$jobId" | ConvertFrom-Json
  $status = $job.status
  Write-Host "status: $status stage: $($job.current_stage)"
  if ($status -eq "completed" -or $status -eq "failed") { break }
}

if ($status -ne "completed") {
  throw "Job did not complete. Final status: $status"
}

$csv = curl.exe -s "http://localhost:8000/jobs/$jobId/result.csv"
$header = ($csv -split "`n")[0].Trim()
$columnCount = ($header -split ',').Count
Write-Host "CSV columns: $columnCount"
Write-Host $header
