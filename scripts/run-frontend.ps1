# Runs the Angular dev server, which uses the local API and AI service URLs from environment.ts.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path (Split-Path -Parent $PSScriptRoot) '..\influencerBrand-mvp-frontend')
& node node_modules\@angular\cli\bin\ng.js serve --port 4200
