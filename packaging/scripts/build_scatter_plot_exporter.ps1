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

$spec = Join-Path $SpecRoot "Scatter Plot Exporter.spec"
if (-not (Test-Path -LiteralPath $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $RepoRoot "dist\Scatter-Plot-Exporter"
$workRoot = Join-Path $RepoRoot "build\scatter_plot_exporter"

Push-Location -LiteralPath $RepoRoot
try {
    $buildArguments = @{
        PythonPath = $python
        SpecPath = $spec
        VersionInfoPath = (Join-Path $RepoRoot "build\version_info\Scatter Plot Exporter.txt")
        ProductName = "Scatter Plot Exporter"
        ExecutableName = "Scatter-Plot-Exporter.exe"
        Version = "1.0.0"
        DistPath = $distRoot
        WorkPath = $workRoot
    }
    Invoke-PyInstallerBuild @buildArguments
} finally {
    Pop-Location
}

Write-Host "Built standalone executable: dist\Scatter-Plot-Exporter\export_scatter_plots.exe"
