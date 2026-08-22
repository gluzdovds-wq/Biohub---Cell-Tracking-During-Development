param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("44b6", "6bba")]
    [string]$Embryo
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$settings = @{
    "44b6" = @{
        Kernel = "dmitriigluzdov/biohub-exp011-audit-loeo-44b6"
        Directory = "kaggle_notebooks\exp011_audit_loeo_44b6"
    }
    "6bba" = @{
        Kernel = "dmitriigluzdov/biohub-exp012-audit-loeo-6bba"
        Directory = "kaggle_notebooks\exp012_audit_loeo_6bba"
    }
}
$audit = $settings[$Embryo]

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
Write-Host $launchedStatus
