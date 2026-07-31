param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version
)

$ErrorActionPreference = "Stop"

$buildScripts = @(
    "build_ng_tile_area_tool.ps1",
    "build_pattern_grid_tile_exporter.ps1",
    "build_matrix_summary_exporter.ps1",
    "build_scatter_plot_exporter.ps1"
)

foreach ($buildScript in $buildScripts) {
    $scriptPath = Join-Path $PSScriptRoot $buildScript
    if (-not (Test-Path -LiteralPath $scriptPath)) {
        throw "Build script not found: $scriptPath"
    }
    & $scriptPath
}

$bundleName = "VisionFlow-Utility-Tools-v$Version-windows-x64"
$bundleRoot = Join-Path $PSScriptRoot "dist\$bundleName"
$zipPath = Join-Path $PSScriptRoot "$bundleName.zip"
if (Test-Path -LiteralPath $bundleRoot) {
    throw "Bundle directory already exists: $bundleRoot"
}
if (Test-Path -LiteralPath $zipPath) {
    throw "Release ZIP already exists: $zipPath"
}

$tools = @(
    @{
        Source = "dist\NG-Tile-Area-Tool\*.exe"
        Name = "NG-Tile-Area-Tool.exe"
    },
    @{
        Source = "dist\Pattern-Grid-Tile-Exporter\export_pattern_grid_tiles.exe"
        Name = "Pattern-Grid-Tile-Exporter.exe"
    },
    @{
        Source = "dist\Matrix-Summary-Exporter\export_matrix_summary.exe"
        Name = "Matrix-Summary-Exporter.exe"
    },
    @{
        Source = "dist\Scatter-Plot-Exporter\export_scatter_plots.exe"
        Name = "Scatter-Plot-Exporter.exe"
    }
)

$resolvedTools = foreach ($tool in $tools) {
    $sourceMatches = @(Get-ChildItem -Path (Join-Path $PSScriptRoot $tool.Source) -File)
    if ($sourceMatches.Count -ne 1) {
        throw "Expected exactly one built executable for $($tool.Source), found $($sourceMatches.Count)"
    }
    @{
        Source = $sourceMatches[0].FullName
        Name = $tool.Name
    }
}

New-Item -ItemType Directory -Path $bundleRoot | Out-Null
foreach ($tool in $resolvedTools) {
    Copy-Item -LiteralPath $tool.Source -Destination (Join-Path $bundleRoot $tool.Name)
}

$readme = Join-Path $PSScriptRoot "UTILITY_TOOLS_README.txt"
if (-not (Test-Path -LiteralPath $readme)) {
    throw "Utility README not found: $readme"
}
Copy-Item -LiteralPath $readme -Destination (Join-Path $bundleRoot "README.txt")
Set-Content -LiteralPath (Join-Path $bundleRoot "VERSION.txt") -Value $Version -Encoding ascii

Compress-Archive -LiteralPath $bundleRoot -DestinationPath $zipPath -CompressionLevel Optimal
$zip = Get-Item -LiteralPath $zipPath
$sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()

Write-Host "Built utility release bundle: $($zip.FullName)"
Write-Host "Size: $($zip.Length) bytes"
Write-Host "SHA-256: $sha256"
