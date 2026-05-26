<#
.SYNOPSIS
    Recursive locator helper for postflight artifact presence checks.

.DESCRIPTION
    Closes anomaly A8 from Run 14 postflight session (2026-05-26).
    Postflight gates must search recursively because critical artifacts may
    live one directory deeper than top-level. Example from Run 14:
      ensemble.joblib lives at      outputs/run14/full/models/ensemble.joblib
      NOT at                        outputs/run14/full/ensemble.joblib

    Fixed Test-Path checks on flat paths return false and trigger spurious
    gate FAILures even when data is fully present. This helper searches
    recursively from a root, optionally enforcing a minimum-byte threshold.

    USAGE — dot-source from any postflight script:
        . "$PSScriptRoot\Test-ArtifactPresent.ps1"
        if (Test-ArtifactPresent -Root $localOutDir -Filename "ensemble.joblib" -MinBytes 1000000) { ... }

    Run 15 prep will dot-source this from Run_Postflight.ps1 and switch
    all gate logic to use this helper instead of fixed Test-Path.
#>

function Test-ArtifactPresent {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Filename,
        [long]$MinBytes = 0
    )
    if (-not (Test-Path $Root)) { return $false }
    $hit = Get-ChildItem -Recurse -LiteralPath $Root -Filter $Filename -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $hit) { return $false }
    if ($MinBytes -gt 0 -and $hit.Length -lt $MinBytes) { return $false }
    return $true
}
