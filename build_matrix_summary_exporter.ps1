$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $PSScriptRoot "Matrix Summary Exporter.spec"
if (-not (Test-Path -LiteralPath $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $PSScriptRoot "dist\Matrix-Summary-Exporter"
$workRoot = Join-Path $PSScriptRoot "build\matrix_summary_exporter"

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

Write-Host "Built standalone executable: dist\Matrix-Summary-Exporter\export_matrix_summary.exe"
