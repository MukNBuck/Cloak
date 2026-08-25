Set-Location $PSScriptRoot
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  Write-Error "Node.js/npm is required for the Cloak desktop app."
  exit 1
}
if (-not (Test-Path "node_modules")) {
  npm install
}
npm start
