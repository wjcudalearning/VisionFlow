$ErrorActionPreference = "Stop"

# Build scripts live in packaging\scripts; the repository root is two levels up.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SpecRoot = Join-Path $RepoRoot "packaging\specs"
. (Join-Path $PSScriptRoot "pyinstaller_path_guard.ps1")

$python = Join-Path $RepoRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $SpecRoot "Pattern Grid Tile Exporter.spec"
if (-not (Test-Path $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $RepoRoot "dist\Pattern-Grid-Tile-Exporter"
$workRoot = Join-Path $RepoRoot "build\pattern_grid_tile_exporter"

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
} finally {
    Pop-Location
}

Write-Host "Built standalone executable: dist\Pattern-Grid-Tile-Exporter\export_pattern_grid_tiles.exe"
