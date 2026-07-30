param(
    [string]$Architecture = "sm_86",
    [string]$BuildDirectory = "",
    [switch]$RunTests,
    [string]$Image = "",
    [string]$Recipe = "",
    [int]$Benchmark = 20
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    Write-Host "> $Executable $($Arguments -join ' ')"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage Exit code: $LASTEXITCODE"
    }
}

function Get-OptionalProperty {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        $DefaultValue = $null
    )

    if ($null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name) {
        return $Object.$Name
    }
    return $DefaultValue
}

if ($Architecture -notmatch '^sm_\d{2,3}$') {
    throw "Invalid architecture '$Architecture'. Expected sm_86, sm_89, etc."
}
if ([bool]$Image -ne [bool]$Recipe) {
    throw "-Image and -Recipe must be provided together."
}

$projectDirectory = $PSScriptRoot
$projectName = Split-Path -Leaf $projectDirectory
$repositoryRoot = Split-Path -Parent $projectDirectory

if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $repositoryRoot "outputs_validation\cuda_build_stage"
}
elseif (-not [System.IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repositoryRoot $BuildDirectory
}

$manifestPath = Join-Path $projectDirectory "cuda_project.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing CUDA project manifest: $manifestPath"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ((Get-OptionalProperty -Object $manifest -Name "schema_version" -DefaultValue 0) -ne 1) {
    throw "Unsupported cuda_project.json schema_version. Expected 1."
}

$outputName = [string](Get-OptionalProperty -Object $manifest -Name "output_name" -DefaultValue $projectName)
$dllSources = @(Get-OptionalProperty -Object $manifest -Name "dll_sources" -DefaultValue @())
$testTargets = @(Get-OptionalProperty -Object $manifest -Name "test_targets" -DefaultValue @())
$includeDirs = @(".") + @(Get-OptionalProperty -Object $manifest -Name "include_dirs" -DefaultValue @())
$nvccConfig = Get-OptionalProperty -Object $manifest -Name "nvcc" -DefaultValue $null

if ($dllSources.Count -eq 0) {
    throw "cuda_project.json must declare at least one dll_sources entry."
}

$nvcc = (Get-Command nvcc -ErrorAction Stop).Source
$dumpbin = (Get-Command dumpbin -ErrorAction Stop).Source
$cl = (Get-Command cl -ErrorAction Stop).Source
$workspacePython = Join-Path $repositoryRoot "env\Scripts\python.exe"
if (Test-Path -LiteralPath $workspacePython -PathType Leaf) {
    $python = (Resolve-Path -LiteralPath $workspacePython).Path
}
else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
$buildPath = (Resolve-Path -LiteralPath $BuildDirectory).Path

$preflightOutput = Join-Path $buildPath "cuda_build_preflight.json"
$preflightScript = Join-Path $projectDirectory "preflight_cuda_build.py"
if (Test-Path -LiteralPath $preflightScript -PathType Leaf) {
    Invoke-CheckedCommand -Executable $python -Arguments @(
        $preflightScript,
        "--output", $preflightOutput
    ) -FailureMessage "CUDA static preflight failed."
}

$resolvedDllSources = foreach ($relativeSource in $dllSources) {
    $candidate = Join-Path $projectDirectory ([string]$relativeSource)
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Missing DLL source declared in cuda_project.json: $candidate"
    }
    (Resolve-Path -LiteralPath $candidate).Path
}

$resolvedIncludeDirs = @()
foreach ($relativeInclude in ($includeDirs | Select-Object -Unique)) {
    $candidate = Join-Path $projectDirectory ([string]$relativeInclude)
    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        throw "Missing include directory declared in cuda_project.json: $candidate"
    }
    $resolvedIncludeDirs += (Resolve-Path -LiteralPath $candidate).Path
}

$dllPath = Join-Path $buildPath "$outputName.dll"
$importLibraryPath = Join-Path $buildPath "$outputName.lib"
$expPath = Join-Path $buildPath "$outputName.exp"
foreach ($artifact in @($dllPath, $importLibraryPath, $expPath)) {
    if (Test-Path -LiteralPath $artifact) {
        Remove-Item -LiteralPath $artifact -Force
    }
}

$optimization = [string](Get-OptionalProperty -Object $nvccConfig -Name "dll_optimization" -DefaultValue "-O2")
$cudart = [string](Get-OptionalProperty -Object $nvccConfig -Name "cudart" -DefaultValue "static")
$testOptimization = [string](Get-OptionalProperty -Object $nvccConfig -Name "test_optimization" -DefaultValue "-O2")

$dllArguments = @(
    "--std=c++17",
    $optimization,
    "--shared",
    "--cudart=$cudart",
    "-arch=$Architecture",
    "-Xcompiler=/MD,/utf-8",
    "-Xlinker", "/IMPLIB:$importLibraryPath"
)
foreach ($includeDir in $resolvedIncludeDirs) {
    $dllArguments += "-I$includeDir"
}
$dllArguments += @("-o", $dllPath)
$dllArguments += $resolvedDllSources

Invoke-CheckedCommand -Executable $nvcc -Arguments $dllArguments -FailureMessage "CUDA DLL build failed."
if (-not (Test-Path -LiteralPath $dllPath -PathType Leaf)) {
    throw "nvcc returned success but did not create $dllPath"
}
if (-not (Test-Path -LiteralPath $importLibraryPath -PathType Leaf)) {
    throw "nvcc returned success but did not create $importLibraryPath"
}

foreach ($target in $testTargets) {
    $targetName = [string](Get-OptionalProperty -Object $target -Name "name" -DefaultValue "")
    if ([string]::IsNullOrWhiteSpace($targetName)) {
        throw "Every test target must declare a non-empty name."
    }

    $targetSources = @(Get-OptionalProperty -Object $target -Name "sources" -DefaultValue @())
    if ($targetSources.Count -eq 0) {
        throw "Test target '$targetName' has no sources."
    }

    $resolvedTargetSources = foreach ($relativeSource in $targetSources) {
        $candidate = Join-Path $projectDirectory ([string]$relativeSource)
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Missing test source declared in cuda_project.json: $candidate"
        }
        (Resolve-Path -LiteralPath $candidate).Path
    }

    $testExe = Join-Path $buildPath "$targetName.exe"
    $testArguments = @(
        "--std=c++17",
        $testOptimization,
        "--cudart=$cudart",
        "-arch=$Architecture",
        "-Xcompiler=/MD,/utf-8"
    )
    foreach ($includeDir in $resolvedIncludeDirs) {
        $testArguments += "-I$includeDir"
    }
    $testArguments += @("-o", $testExe)
    $testArguments += $resolvedTargetSources
    $testArguments += $importLibraryPath

    Invoke-CheckedCommand -Executable $nvcc -Arguments $testArguments -FailureMessage "CUDA test target '$targetName' failed to compile."
}

$exportsText = & $dumpbin /exports $dllPath | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /exports failed with exit code $LASTEXITCODE"
}
$exportsText | Set-Content -LiteralPath (Join-Path $buildPath "exports.txt") -Encoding UTF8

if (Test-Path -LiteralPath $preflightOutput) {
    $expectedExports = @((Get-Content -LiteralPath $preflightOutput -Raw | ConvertFrom-Json).exports)
    foreach ($expectedExport in $expectedExports) {
        if ($exportsText -notmatch "(?m)\b$([regex]::Escape([string]$expectedExport))\b") {
            throw "CUDA DLL is missing expected export: $expectedExport"
        }
    }
}

$dependenciesText = & $dumpbin /dependents $dllPath | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /dependents failed with exit code $LASTEXITCODE"
}
$dependenciesText | Set-Content -LiteralPath (Join-Path $buildPath "dependencies.txt") -Encoding UTF8

Write-Host "Built DLL: $dllPath"
Write-Host "Built import library: $importLibraryPath"
Get-ChildItem -LiteralPath $buildPath | ForEach-Object {
    Write-Host "Build output: $($_.Name)"
}

$publishArtifacts = @(
    $dllPath,
    $importLibraryPath,
    $expPath
)
foreach ($target in $testTargets) {
    $publishArtifacts += Join-Path $buildPath "$([string]$target.name).exe"
}
foreach ($artifact in $publishArtifacts) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        throw "Required CUDA build artifact is missing: $artifact"
    }
    Copy-Item -LiteralPath $artifact -Destination (Join-Path $projectDirectory (Split-Path -Leaf $artifact)) -Force
}
Write-Host "Published validated CUDA artifacts to: $projectDirectory"

$artifactHashes = foreach ($artifact in $publishArtifacts) {
    $hash = Get-FileHash -LiteralPath $artifact -Algorithm SHA256
    [pscustomobject]@{
        name = Split-Path -Leaf $artifact
        bytes = (Get-Item -LiteralPath $artifact).Length
        sha256 = $hash.Hash
    }
}
$commit = (& git -C $repositoryRoot rev-parse HEAD 2>$null | Select-Object -First 1)
$buildEvidence = [pscustomobject]@{
    schema_version = 1
    architecture = $Architecture
    commit = [string]$commit
    nvcc_path = $nvcc
    nvcc_version = ((& $nvcc --version | Out-String).Trim())
    cl_path = $cl
    cl_version = (Get-Item -LiteralPath $cl).VersionInfo.FileVersion
    dumpbin_path = $dumpbin
    dumpbin_version = (Get-Item -LiteralPath $dumpbin).VersionInfo.FileVersion
    artifacts = @($artifactHashes)
}
$buildEvidence |
    ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $buildPath "cuda_build_evidence.json") -Encoding UTF8
Write-Host "Build evidence: $(Join-Path $buildPath 'cuda_build_evidence.json')"

if ($RunTests) {
    foreach ($target in $testTargets) {
        $publishedTest = Join-Path $projectDirectory "$([string]$target.name).exe"
        Invoke-CheckedCommand -Executable $publishedTest -Arguments @() `
            -FailureMessage "CUDA test target '$([string]$target.name)' failed."
    }

    $validatorArguments = @(
        (Join-Path $projectDirectory "validate_cuda_dll.py"),
        "--dll", (Join-Path $projectDirectory "$outputName.dll"),
        "--benchmark", [string]$Benchmark
    )
    if ($Image -and $Recipe) {
        $validatorArguments += @("--image", $Image, "--recipe", $Recipe)
    }
    Invoke-CheckedCommand -Executable $python -Arguments $validatorArguments `
        -FailureMessage "Python CUDA validation failed."
}
