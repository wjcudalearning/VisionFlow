$ErrorActionPreference = "Stop"

# Build scripts live in packaging\scripts; the repository root is two levels up.
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$SpecRoot = Join-Path $RepoRoot "packaging\specs"
. (Join-Path $PSScriptRoot "pyinstaller_path_guard.ps1")
. (Join-Path $PSScriptRoot "pyinstaller_build.ps1")

$python = Join-Path $RepoRoot "env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $SpecRoot "Matrix Summary Exporter.spec"
if (-not (Test-Path -LiteralPath $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $RepoRoot "dist\Matrix-Summary-Exporter"
$workRoot = Join-Path $RepoRoot "build\matrix_summary_exporter"

Push-Location -LiteralPath $RepoRoot
try {
    $buildArguments = @{
        PythonPath = $python
        SpecPath = $spec
        VersionInfoPath = (Join-Path $RepoRoot "build\version_info\Matrix Summary Exporter.txt")
        ProductName = "Matrix Summary Exporter"
        ExecutableName = "Matrix-Summary-Exporter.exe"
        Version = "1.0.0"
        DistPath = $distRoot
        WorkPath = $workRoot
    }
    Invoke-PyInstallerBuild @buildArguments
} finally {
    Pop-Location
}

Write-Host "Built standalone executable: dist\Matrix-Summary-Exporter\export_matrix_summary.exe"
