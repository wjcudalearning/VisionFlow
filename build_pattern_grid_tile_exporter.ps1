$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $PSScriptRoot "Pattern Grid Tile Exporter.spec"
if (-not (Test-Path $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $PSScriptRoot "dist\Pattern-Grid-Tile-Exporter"
$workRoot = Join-Path $PSScriptRoot "build\pattern_grid_tile_exporter"

Push-Location $PSScriptRoot
try {
    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $distRoot `
        --workpath $workRoot `
        $spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Built standalone executable: dist\Pattern-Grid-Tile-Exporter\export_pattern_grid_tiles.exe"
