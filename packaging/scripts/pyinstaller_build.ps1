# Dot-source after pyinstaller_path_guard.ps1.

$script:PyInstallerBuildHelperRoot = $PSScriptRoot

function Invoke-PyInstallerBuild {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$SpecPath,
        [Parameter(Mandatory = $true)][string]$VersionInfoPath,
        [Parameter(Mandatory = $true)][string]$ProductName,
        [Parameter(Mandatory = $true)][string]$ExecutableName,
        [Parameter(Mandatory = $true)][string]$Version,
        [string]$DistPath = "",
        [string]$WorkPath = ""
    )

    foreach ($requiredPath in @($PythonPath, $SpecPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required PyInstaller input not found: $requiredPath"
        }
    }
    $environmentCheck = [System.IO.Path]::GetFullPath(
        (Join-Path $script:PyInstallerBuildHelperRoot "..\..\tools\requirements_lock.py")
    )
    if (-not (Test-Path -LiteralPath $environmentCheck -PathType Leaf)) {
        throw "Build environment validator not found: $environmentCheck"
    }
    & $PythonPath $environmentCheck --check-environment
    if ($LASTEXITCODE -ne 0) {
        throw "Build Python does not match requirements.lock.txt"
    }

    $versionWriter = Join-Path $script:PyInstallerBuildHelperRoot "write_version_info.py"
    if (-not (Test-Path -LiteralPath $versionWriter -PathType Leaf)) {
        throw "Version resource generator not found: $versionWriter"
    }

    $versionArguments = @(
        $versionWriter,
        "--output", $VersionInfoPath,
        "--product-name", $ProductName,
        "--executable-name", $ExecutableName,
        "--version", $Version
    )
    & $PythonPath @versionArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Version resource generation failed with exit code $LASTEXITCODE"
    }

    $pyInstallerArguments = @("-m", "PyInstaller", "--noconfirm", "--clean")
    if ($DistPath) { $pyInstallerArguments += @("--distpath", $DistPath) }
    if ($WorkPath) { $pyInstallerArguments += @("--workpath", $WorkPath) }
    $pyInstallerArguments += $SpecPath

    Invoke-WithCleanBuildPath {
        & $PythonPath @pyInstallerArguments
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE"
        }
    }
}
