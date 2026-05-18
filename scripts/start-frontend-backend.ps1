$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "frontend")

$env:REACT_APP_API_URL = "http://localhost:8000"
$env:REACT_APP_RECOGNITION_MODE = "backend"
$env:DISABLE_ESLINT_PLUGIN = "true"
npm.cmd start
