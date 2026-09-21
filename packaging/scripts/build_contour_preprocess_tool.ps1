param(
    [string]$Version = "1.1.0",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"

# Build scripts live in packaging\scripts; the repository root is two levels up.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SpecRoot = Join-Path $RepoRoot "packaging\specs"

$expectedVersion = "1.1.0"
if ($Version -ne $expectedVersion) {
    throw "Requested version $Version does not match source version $expectedVersion"
}

$python = Join-Path $RepoRoot "env\Scripts\python.exe"
$spec = Join-Path $SpecRoot "Traditional CV Tuning Tool.spec"
$readme = Join-Path $RepoRoot "contour_preprocess_tool\README.md"
$distRoot = if ($OutputDirectory) {
    [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDirectory))
} else {
    Join-Path $RepoRoot "dist\Traditional-CV-Tuning-Tool"
}
$workRoot = Join-Path $RepoRoot "build\traditional_cv_tuning_tool"
$exePath = Join-Path $distRoot "Traditional CV Tuning Tool.exe"

foreach ($requiredPath in @($python, $spec, $readme)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required build input not found: $requiredPath"
    }
}
if (Test-Path -LiteralPath $exePath) {
    throw "Refusing to overwrite an existing versioned executable: $exePath"
}

Push-Location $RepoRoot
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

    Copy-Item -LiteralPath $readme -Destination (Join-Path $distRoot "README.md") -Force
    $commit = (& git rev-parse HEAD).Trim()
    @(
        "Traditional CV Tuning Tool"
        "Version: $Version"
        "Git commit: $commit"
        "Platform: Windows x64"
        "Processing: CPU / OpenCV"
        "Preview: Qt OpenGL when available, raster fallback"
    ) | Set-Content -LiteralPath (Join-Path $distRoot "VERSION.txt") -Encoding UTF8
} finally {
    Pop-Location
}

Write-Host "Built standalone tool in $distRoot"
