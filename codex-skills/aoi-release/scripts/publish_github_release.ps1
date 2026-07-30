[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Repository,

    [Parameter(Mandatory = $true)]
    [string]$Tag,

    [Parameter(Mandatory = $true)]
    [string]$AssetPath,

    [Parameter(Mandatory = $true)]
    [string]$ReleaseName,

    [Parameter(Mandatory = $true)]
    [string]$BodyPath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$ExpectedCommit,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedSha256,

    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$apiVersion = '2022-11-28'
$apiBase = "https://api.github.com/repos/$Repository"

function Get-HttpStatusCode {
    param([Parameter(Mandatory = $true)]$ErrorRecord)

    $response = $ErrorRecord.Exception.Response
    if ($null -eq $response -or $null -eq $response.StatusCode) {
        return $null
    }
    return [int]$response.StatusCode
}

function Get-GitHubCredentialHeaders {
    $credentialLines = "protocol=https`nhost=github.com`n`n" | git credential fill
    if ($LASTEXITCODE -ne 0) {
        throw 'git credential fill failed for github.com'
    }

    $credential = @{}
    foreach ($line in $credentialLines) {
        $parts = $line -split '=', 2
        if ($parts.Count -eq 2) {
            $credential[$parts[0]] = $parts[1]
        }
    }
    if (-not $credential.ContainsKey('password')) {
        throw 'GitHub credential is unavailable'
    }

    return @{
        Authorization = "Bearer $($credential['password'])"
        Accept = 'application/vnd.github+json'
        'X-GitHub-Api-Version' = $apiVersion
    }
}

$resolvedAsset = (Resolve-Path -LiteralPath $AssetPath).Path
$resolvedBody = (Resolve-Path -LiteralPath $BodyPath).Path
$asset = Get-Item -LiteralPath $resolvedAsset
if (-not $asset.PSIsContainer -and $asset.Length -le 0) {
    throw "Release asset is empty: $resolvedAsset"
}
if ($asset.PSIsContainer) {
    throw "Release asset must be a file: $resolvedAsset"
}

$actualSha256 = (Get-FileHash -LiteralPath $resolvedAsset -Algorithm SHA256).Hash
if ($actualSha256 -ne $ExpectedSha256.ToUpperInvariant()) {
    throw "Release asset SHA-256 mismatch: $actualSha256"
}

$localTagCommit = (& git rev-parse "$Tag^{}").Trim()
if ($LASTEXITCODE -ne 0 -or $localTagCommit -ne $ExpectedCommit.ToLowerInvariant()) {
    throw "Local tag does not resolve to expected commit: $localTagCommit"
}

& git merge-base --is-ancestor $ExpectedCommit origin/main
if ($LASTEXITCODE -ne 0) {
    throw "Expected release commit is not contained in origin/main: $ExpectedCommit"
}

$remoteTagLines = @(& git ls-remote --tags origin "refs/tags/$Tag" "refs/tags/$Tag^{}")
if ($LASTEXITCODE -ne 0 -or $remoteTagLines.Count -eq 0) {
    throw "Remote tag is missing: $Tag"
}
$peeledLine = @($remoteTagLines | Where-Object { $_ -match "refs/tags/$([regex]::Escape($Tag))\^\{\}$" })
$remoteCommit = if ($peeledLine.Count -eq 1) {
    ($peeledLine[0] -split '\s+')[0]
} else {
    ($remoteTagLines[0] -split '\s+')[0]
}
if ($remoteCommit -ne $ExpectedCommit.ToLowerInvariant()) {
    throw "Remote tag does not resolve to expected commit: $remoteCommit"
}

$headers = Get-GitHubCredentialHeaders
$existingRelease = $null
try {
    $existingRelease = Invoke-RestMethod -Method Get -Uri "$apiBase/releases/tags/$Tag" -Headers $headers
} catch {
    if ((Get-HttpStatusCode -ErrorRecord $_) -ne 404) {
        throw
    }
}

$preflight = [ordered]@{
    Repository = $Repository
    Tag = $Tag
    ExpectedCommit = $ExpectedCommit.ToLowerInvariant()
    RemoteCommit = $remoteCommit
    AssetName = $asset.Name
    AssetBytes = $asset.Length
    AssetSha256 = $actualSha256
    ReleaseExists = ($null -ne $existingRelease)
    ExistingReleaseUrl = if ($null -ne $existingRelease) { $existingRelease.html_url } else { $null }
    CredentialAvailable = $true
}

if ($PreflightOnly) {
    [pscustomobject]$preflight
    return
}
if ($null -ne $existingRelease) {
    throw "Release already exists: $($existingRelease.html_url)"
}

$body = Get-Content -LiteralPath $resolvedBody -Raw -Encoding utf8
$createPayload = @{
    tag_name = $Tag
    target_commitish = $ExpectedCommit
    name = $ReleaseName
    body = $body
    draft = $true
    prerelease = $false
} | ConvertTo-Json -Depth 5

$release = Invoke-RestMethod -Method Post -Uri "$apiBase/releases" -Headers $headers -Body $createPayload -ContentType 'application/json; charset=utf-8'
try {
    $uploadBase = $release.upload_url -replace '\{\?name,label\}$', ''
    $escapedAssetName = [System.Uri]::EscapeDataString($asset.Name)
    $uploadedAsset = Invoke-RestMethod -Method Post -Uri "${uploadBase}?name=$escapedAssetName" -Headers $headers -InFile $resolvedAsset -ContentType 'application/zip'
    if ($uploadedAsset.size -ne $asset.Length) {
        throw "Uploaded asset size mismatch: $($uploadedAsset.size)"
    }

    $publishPayload = @{
        draft = $false
        prerelease = $false
        make_latest = 'true'
    } | ConvertTo-Json
    $publishedRelease = Invoke-RestMethod -Method Patch -Uri "$apiBase/releases/$($release.id)" -Headers $headers -Body $publishPayload -ContentType 'application/json'
} catch {
    throw "Release upload failed; draft retained at $($release.html_url): $($_.Exception.Message)"
}

$verifiedRelease = Invoke-RestMethod -Method Get -Uri "$apiBase/releases/tags/$Tag" -Headers $headers
$matchingAssets = @($verifiedRelease.assets | Where-Object { $_.name -eq $asset.Name })
if ($verifiedRelease.draft -or $verifiedRelease.prerelease -or $matchingAssets.Count -ne 1) {
    throw 'Published release metadata verification failed'
}
if ($matchingAssets[0].size -ne $asset.Length) {
    throw "Published asset size mismatch: $($matchingAssets[0].size)"
}

$downloadPath = Join-Path ([IO.Path]::GetTempPath()) ("aoi-release-" + [guid]::NewGuid().ToString('N') + '.zip')
try {
    Invoke-WebRequest -Uri $matchingAssets[0].browser_download_url -OutFile $downloadPath
    $downloadedAsset = Get-Item -LiteralPath $downloadPath
    $downloadedSha256 = (Get-FileHash -LiteralPath $downloadPath -Algorithm SHA256).Hash
    if ($downloadedAsset.Length -ne $asset.Length) {
        throw "Downloaded asset size mismatch: $($downloadedAsset.Length)"
    }
    if ($downloadedSha256 -ne $actualSha256) {
        throw "Downloaded asset SHA-256 mismatch: $downloadedSha256"
    }
} finally {
    Remove-Item -LiteralPath $downloadPath -Force -ErrorAction SilentlyContinue
}

[pscustomobject]@{
    ReleaseUrl = $verifiedRelease.html_url
    DirectDownloadUrl = $matchingAssets[0].browser_download_url
    Tag = $verifiedRelease.tag_name
    Draft = $verifiedRelease.draft
    Prerelease = $verifiedRelease.prerelease
    AssetName = $matchingAssets[0].name
    AssetBytes = $matchingAssets[0].size
    Sha256 = $actualSha256
    Commit = $remoteCommit
}
