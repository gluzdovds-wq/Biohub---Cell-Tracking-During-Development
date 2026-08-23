param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("44b6", "6bba")]
    [string]$Embryo
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$settings = @{
    "44b6" = @{
        Upstream = "dmitriigluzdov/biohub-exp011-audit-loeo-44b6"
        Kernel = "dmitriigluzdov/biohub-exp050-weak-tie-break-44b6"
        Directory = "kaggle_notebooks\exp050_weak_tiebreak_44b6"
        Source = "kaggle_notebooks\exp050_weak_tiebreak_44b6\weak_tiebreak_44b6.py"
        SourceSha = "7c24821a12baf64f82df5ce0f59d2b9f24ccdb2d0f7e62819540c897577b76d2"
    }
    "6bba" = @{
        Upstream = "dmitriigluzdov/biohub-exp012-audit-loeo-6bba"
        Kernel = "dmitriigluzdov/biohub-exp051-weak-tie-break-6bba"
        Directory = "kaggle_notebooks\exp051_weak_tiebreak_6bba"
        Source = "kaggle_notebooks\exp051_weak_tiebreak_6bba\weak_tiebreak_6bba.py"
        SourceSha = "37492dec18d8e2cbbb171171034af9b4db7bd58ff350d9a9aa81b97db471ca50"
    }
}
$audit = $settings[$Embryo]

$sourcePath = Join-Path $repoRoot $audit.Source
$sourceSha = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sourceSha -ne $audit.SourceSha) {
    throw "Weak tie-break source SHA drift for ${Embryo}: $sourceSha"
}

$upstreamText = (& kaggle kernels status $audit.Upstream 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $upstreamText -notmatch 'KernelWorkerStatus\.COMPLETE') {
    throw "Upstream nested audit is not complete for ${Embryo}: $upstreamText"
}

$statusText = (& kaggle kernels status $audit.Kernel 2>&1 | Out-String).Trim()
$statusExitCode = $LASTEXITCODE
if ($statusExitCode -eq 0 -and $statusText -match 'KernelWorkerStatus\.(QUEUED|RUNNING|COMPLETE)') {
    Write-Host "Weak tie-break audit already active or complete: $statusText"
    exit 0
}

$auditDirectory = Join-Path $repoRoot $audit.Directory
& kaggle kernels push -p $auditDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Failed to push weak tie-break audit $($audit.Kernel)"
}

$launchedText = (& kaggle kernels status $audit.Kernel 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $launchedText -notmatch 'KernelWorkerStatus\.(QUEUED|RUNNING)') {
    throw "Unexpected status after weak tie-break push: $launchedText"
}
Write-Host "H050 weak tie-break audit launch: PASS"
Write-Host "Source SHA: PASS ($sourceSha)"
Write-Host $launchedText
