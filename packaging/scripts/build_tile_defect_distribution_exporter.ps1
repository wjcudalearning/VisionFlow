$ErrorActionPreference = "Stop"

# Build scripts live in packaging\scripts; the repository root is two levels up.
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$SpecRoot = Join-Path $RepoRoot "packaging\specs"
. (Join-Path $PSScriptRoot "pyinstaller_path_guard.ps1")
. (Join-Path $PSScriptRoot "pyinstaller_build.ps1")

$python = Join-Path $RepoRoot "env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $SpecRoot "Tile Defect Distribution Exporter.spec"
if (-not (Test-Path -LiteralPath $spec -PathType Leaf)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $RepoRoot "dist\Tile-Defect-Distribution-Exporter"
$workRoot = Join-Path $RepoRoot "build\tile_defect_distribution_exporter"

Push-Location -LiteralPath $RepoRoot
try {
    $buildArguments = @{
        PythonPath = $python
        SpecPath = $spec
        VersionInfoPath = (Join-Path $RepoRoot "build\version_info\Tile Defect Distribution Exporter.txt")
        ProductName = "Tile Defect Distribution Exporter"
        ExecutableName = "Tile-Defect-Distribution-Exporter.exe"
        Version = "1.1.0"
        DistPath = $distRoot
        WorkPath = $workRoot
    }
    Invoke-PyInstallerBuild @buildArguments
} finally {
    Pop-Location
}

Write-Host "Built standalone executable: dist\Tile-Defect-Distribution-Exporter\export_tile_defect_distribution.exe"
