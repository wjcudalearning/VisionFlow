$ErrorActionPreference = "Stop"

# Build scripts live in packaging\scripts; the repository root is two levels up.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SpecRoot = Join-Path $RepoRoot "packaging\specs"
. (Join-Path $PSScriptRoot "pyinstaller_path_guard.ps1")

$python = Join-Path $RepoRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $SpecRoot "NG Tile Area Tool.spec"
if (-not (Test-Path $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $RepoRoot "dist\NG-Tile-Area-Tool"
$workRoot = Join-Path $RepoRoot "build\ng_tile_area_tool"
$readme = Join-Path $RepoRoot "docs\packaging\NG_TILE_AREA_TOOL_README.txt"

Push-Location $RepoRoot
try {
    Invoke-WithCleanBuildPath {
        & $python -m PyInstaller `
            --noconfirm `
            --clean `
            --distpath $distRoot `
            --workpath $workRoot `
            $spec
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE"
        }
    }
    Copy-Item -Force $readme $distRoot
} finally {
    Pop-Location
}

Write-Host "Built standalone utility in dist\NG-Tile-Area-Tool"
