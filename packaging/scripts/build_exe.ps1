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

$spec = Join-Path $SpecRoot "VisionFlow AOI.spec"
if (-not (Test-Path -LiteralPath $spec -PathType Leaf)) {
    throw "PyInstaller spec not found: $spec"
}

$cudaDll = Join-Path $RepoRoot "gpu\visionflow_cuda.dll"
if (Test-Path -LiteralPath $cudaDll -PathType Leaf) {
    Write-Host "Including CUDA DLL: $cudaDll"
} else {
    Write-Host "CUDA DLL not found; building CPU-compatible package."
}

Push-Location -LiteralPath $RepoRoot
try {
    $commit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Unable to resolve build commit" }
    $dirty = [bool](& git status --porcelain --untracked-files=no)
    @{ commit = $commit; dirty = $dirty } |
        ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $RepoRoot "build_provenance.json") -Encoding utf8
    $buildArguments = @{
        PythonPath = $python
        SpecPath = $spec
        VersionInfoPath = (Join-Path $RepoRoot "build\version_info\VisionFlow AOI.txt")
        ProductName = "VisionFlow AOI"
        ExecutableName = "VisionFlow AOI.exe"
        Version = "2.0.0"
    }
    Invoke-PyInstallerBuild @buildArguments
} finally {
    Remove-Item -LiteralPath (Join-Path $RepoRoot "build_provenance.json") -Force -ErrorAction SilentlyContinue
    Pop-Location
}

# On-site reference sheets beside the EXE: the camera machine has no network and files only go in.
$distRoot = Join-Path $RepoRoot "dist\VisionFlow AOI"
$siteDocs = @{
    "docs\packaging\DEVICE_PARAMETER_GUIDE.md" = "DEVICE_PARAMETER_GUIDE.md"
    "docs\device-error-codes.md" = "ERROR_CODES.md"
}
foreach ($source in $siteDocs.Keys) {
    $sourcePath = Join-Path $RepoRoot $source
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "On-site document not found: $sourcePath"
    }
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $distRoot $siteDocs[$source]) -Force
}

Write-Host "Built GUI executable: dist\VisionFlow AOI\VisionFlow AOI.exe"
