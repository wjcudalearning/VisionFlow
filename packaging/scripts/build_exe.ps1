$ErrorActionPreference = "Stop"

# Build scripts live in packaging\scripts; the repository root is two levels up.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SpecRoot = Join-Path $RepoRoot "packaging\specs"
. (Join-Path $PSScriptRoot "pyinstaller_path_guard.ps1")

$python = Join-Path $RepoRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $SpecRoot "VisionFlow AOI.spec"
if (-not (Test-Path $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$cudaDll = Join-Path $RepoRoot "gpu\visionflow_cuda.dll"
if (Test-Path $cudaDll) {
    Write-Host "Including CUDA DLL: $cudaDll"
} else {
    Write-Host "CUDA DLL not found; building CPU-compatible package."
}

Push-Location $RepoRoot
try {
    $commit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Unable to resolve build commit" }
    $dirty = [bool](& git status --porcelain --untracked-files=no)
    @{ commit = $commit; dirty = $dirty } |
        ConvertTo-Json |
        Set-Content -Encoding utf8 (Join-Path $RepoRoot "build_provenance.json")
    Invoke-WithCleanBuildPath {
        & $python -m PyInstaller --noconfirm --clean $spec
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE"
        }
    }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $RepoRoot "build_provenance.json")
    Pop-Location
}

Write-Host "Built GUI executable: dist\VisionFlow AOI\VisionFlow AOI.exe"
