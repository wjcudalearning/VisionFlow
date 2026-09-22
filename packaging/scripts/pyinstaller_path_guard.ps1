# Dot-source from build scripts before running PyInstaller.
#
# PyInstaller resolves DLL dependencies through PATH. Agent runtimes can prepend their own
# native bin folders (for example %USERPROFILE%\.cache\codex-runtimes\...\poppler\Library\bin),
# which leaks an unrelated ucrtbase.dll, ICU or OpenSSL into the bundle. The v1.8.0 build
# picked those up and the packaged EXE failed to import QtCore. Such entries are removed only
# while PyInstaller runs; the caller's PATH is restored afterwards.

$script:ForeignBuildPathPatterns = @(
    '\\\.cache\\codex-runtimes(\\|$)'
)

function Get-CleanBuildPath {
    param([string]$PathValue = $env:PATH)

    $kept = New-Object System.Collections.Generic.List[string]
    $removed = New-Object System.Collections.Generic.List[string]
    foreach ($entry in ($PathValue -split ';')) {
        if (-not $entry) { continue }
        $foreign = $false
        foreach ($pattern in $script:ForeignBuildPathPatterns) {
            if ($entry -match $pattern) { $foreign = $true; break }
        }
        if ($foreign) { $removed.Add($entry) } else { $kept.Add($entry) }
    }
    [pscustomobject]@{
        Path = ($kept -join ';')
        Removed = $removed.ToArray()
    }
}

function Invoke-WithCleanBuildPath {
    param([Parameter(Mandatory = $true)][scriptblock]$ScriptBlock)

    $originalPath = $env:PATH
    $clean = Get-CleanBuildPath -PathValue $originalPath
    if ($clean.Removed.Count -gt 0) {
        Write-Host "Excluding $($clean.Removed.Count) agent runtime PATH entries from the PyInstaller build:"
        foreach ($entry in $clean.Removed) { Write-Host "  $entry" }
    }
    $env:PATH = $clean.Path
    try {
        & $ScriptBlock
    } finally {
        $env:PATH = $originalPath
    }
}
