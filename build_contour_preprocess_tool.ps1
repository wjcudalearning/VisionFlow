param(
    [string]$Version = "1.0.0",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"

$expectedVersion = "1.0.0"
if ($Version -ne $expectedVersion) {
    throw "Requested version $Version does not match source version $expectedVersion"
}

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
$spec = Join-Path $PSScriptRoot "Traditional CV Tuning Tool.spec"
$readme = Join-Path $PSScriptRoot "contour_preprocess_tool\README.md"
$distRoot = if ($OutputDirectory) {
    [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $OutputDirectory))
} else {
    Join-Path $PSScriptRoot "dist\Traditional-CV-Tuning-Tool"
}
$workRoot = Join-Path $PSScriptRoot "build\traditional_cv_tuning_tool"
$exePath = Join-Path $distRoot "Traditional CV Tuning Tool.exe"

foreach ($requiredPath in @($python, $spec, $readme)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required build input not found: $requiredPath"
    }
}
if (Test-Path -LiteralPath $exePath) {
    throw "Refusing to overwrite an existing versioned executable: $exePath"
}

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
