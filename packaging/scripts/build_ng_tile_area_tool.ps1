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

$spec = Join-Path $SpecRoot "NG Tile Area Tool.spec"
if (-not (Test-Path -LiteralPath $spec -PathType Leaf)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $RepoRoot "dist\NG-Tile-Area-Tool"
$workRoot = Join-Path $RepoRoot "build\ng_tile_area_tool"
$readme = Join-Path $RepoRoot "docs\packaging\NG_TILE_AREA_TOOL_README.txt"

Push-Location -LiteralPath $RepoRoot
try {
    $buildArguments = @{
        PythonPath = $python
        SpecPath = $spec
        VersionInfoPath = (Join-Path $RepoRoot "build\version_info\NG Tile Area Tool.txt")
        ProductName = "NG Tile Area Tool"
        ExecutableName = "NG-Tile-Area-Tool.exe"
        Version = "1.0.0"
        DistPath = $distRoot
        WorkPath = $workRoot
    }
    Invoke-PyInstallerBuild @buildArguments
    Copy-Item -LiteralPath $readme -Destination $distRoot -Force
} finally {
    Pop-Location
}

Write-Host "Built standalone utility in dist\NG-Tile-Area-Tool"
