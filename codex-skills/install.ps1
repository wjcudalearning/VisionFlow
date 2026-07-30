[CmdletBinding()]
param(
    [string]$CodexHome
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($CodexHome)) {
    if (-not [string]::IsNullOrWhiteSpace($env:CODEX_HOME)) {
        $CodexHome = $env:CODEX_HOME
    }
    else {
        $CodexHome = Join-Path $env:USERPROFILE '.codex'
    }
}

$skillNames = @(
    'aoi-verify-push',
    'aoi-detector-development',
    'aoi-cuda-validate',
    'aoi-release'
)
$destinationRoot = Join-Path $CodexHome 'skills'
$existingSkills = @(
    $skillNames | Where-Object {
        Test-Path -LiteralPath (Join-Path $destinationRoot $_)
    }
)

if ($existingSkills.Count -gt 0) {
    throw "Installation stopped. Existing skills were not overwritten: $($existingSkills -join ', ')"
}

New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null

foreach ($skillName in $skillNames) {
    $source = Join-Path $PSScriptRoot $skillName
    if (-not (Test-Path -LiteralPath (Join-Path $source 'SKILL.md'))) {
        throw "Skill source is missing: $source"
    }

    Copy-Item -Recurse -LiteralPath $source -Destination $destinationRoot
}

Write-Host "Installed $($skillNames.Count) AOI skills to $destinationRoot"
Write-Host 'Restart Codex or start a new session to reload the skills.'
