$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "frontend")

$env:REACT_APP_RECOGNITION_MODE = "mock"
$env:REACT_APP_API_URL = ""
$env:DISABLE_ESLINT_PLUGIN = "true"
npm.cmd start
