param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("44b6", "6bba")]
    [string]$Embryo
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$physicalSmokePath = Join-Path $repoRoot "outputs\exp043_kaggle_v1\exp043_physical_prune_runtime_smoke.json"
$expectedPhysicalSmokeSha = "944f24b5c8fef67ff59bce27f6a447ec4176dafc56203eb2923ca07ebfc5eb71"
$settings = @{
    "44b6" = @{
        Kernel = "dmitriigluzdov/biohub-exp011-audit-loeo-44b6"
        Directory = "kaggle_notebooks\exp011_audit_loeo_44b6"
        Source = "kaggle_notebooks\exp011_audit_loeo_44b6\audit_loeo_44b6.py"
        SourceSha = "61eac88da003d5254eb2a3196cf73f4aba47b175d19af34c86a4911e53dd18df"
    }
    "6bba" = @{
        Kernel = "dmitriigluzdov/biohub-exp012-audit-loeo-6bba"
        Directory = "kaggle_notebooks\exp012_audit_loeo_6bba"
        Source = "kaggle_notebooks\exp012_audit_loeo_6bba\audit_loeo_6bba.py"
        SourceSha = "039bde3e4e4ea138820b17313cddc6754c97b9fb5bd8e9e071253dc32ca7e7bb"
    }
}
$audit = $settings[$Embryo]

if (-not (Test-Path -LiteralPath $physicalSmokePath -PathType Leaf)) {
    throw "Missing EXP043 physical-prune runtime receipt: $physicalSmokePath"
}
$physicalSmokeSha = (Get-FileHash -LiteralPath $physicalSmokePath -Algorithm SHA256).Hash.ToLowerInvariant()
$physicalSmoke = Get-Content -LiteralPath $physicalSmokePath -Raw | ConvertFrom-Json
if ($physicalSmokeSha -ne $expectedPhysicalSmokeSha -or
    $physicalSmoke.status -ne "PASS_PHYSICAL_PRUNE_RUNTIME_CONTRACT" -or
    -not $physicalSmoke.strict_removals_are_subset) {
    throw "EXP043 physical-prune runtime contract failed or drifted: $physicalSmokeSha"
}

$auditSource = Join-Path $repoRoot $audit.Source
$auditSourceSha = (Get-FileHash -LiteralPath $auditSource -Algorithm SHA256).Hash.ToLowerInvariant()
if ($auditSourceSha -ne $audit.SourceSha) {
    throw "Audit source SHA drift for ${Embryo}: $auditSourceSha"
}

& (Join-Path $PSScriptRoot "verify_loeo_parent.ps1") -Embryo $Embryo
if ($LASTEXITCODE -ne 0) {
    throw "Parent verification failed for $Embryo"
}

$statusText = (& kaggle kernels status $audit.Kernel 2>&1 | Out-String).Trim()
$statusExitCode = $LASTEXITCODE
if ($statusExitCode -eq 0 -and $statusText -match 'KernelWorkerStatus\.(QUEUED|RUNNING|COMPLETE)') {
    Write-Host "Audit already active or complete: $statusText"
    exit 0
}

$auditDirectory = Join-Path $repoRoot $audit.Directory
& kaggle kernels push -p $auditDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Failed to push audit kernel $($audit.Kernel)"
}

$launchedStatus = (& kaggle kernels status $audit.Kernel 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Audit was pushed but its status cannot be read: $launchedStatus"
}
if ($launchedStatus -notmatch 'KernelWorkerStatus\.(QUEUED|RUNNING)') {
    throw "Unexpected audit status after push: $launchedStatus"
}
Write-Host "Audit launch: PASS"
Write-Host "EXP043 physical-prune runtime receipt: PASS ($physicalSmokeSha)"
Write-Host "Audit source SHA: PASS ($auditSourceSha)"
Write-Host $launchedStatus
