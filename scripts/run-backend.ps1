# Runs the existing backend jar against the local demo database, from .demo/ so uploads stay out of the backend repo.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
foreach ($line in Get-Content (Join-Path $root '.env')) {
    if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
        $name, $value = $line -split '=', 2
        Set-Item -Path "Env:$name" -Value $value
    }
}
$demo = Join-Path $root '.demo'
New-Item -ItemType Directory -Force $demo | Out-Null
Set-Location $demo
# Windows reports the legacy zone name Asia/Calcutta, which this Postgres rejects at connection time.
& java -Xmx512m '-Duser.timezone=Asia/Kolkata' -jar (Join-Path $root '..\influencerBrand-mvp-backend\target\Influencer-mvp-0.0.1-SNAPSHOT.jar')
