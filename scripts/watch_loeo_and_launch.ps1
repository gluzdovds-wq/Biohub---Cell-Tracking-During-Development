param(
    [ValidateRange(15, 600)]
    [int]$PollSeconds = 60,
    [ValidateRange(1, 12)]
    [int]$MaxHours = 8,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$mutex = [System.Threading.Mutex]::new($false, "Global\BiohubLoeoAuditWatcher")
$hasMutex = $false

function Get-KernelState([string]$Kernel, [switch]$AllowAbsent) {
    $lastText = ""
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $lastText = (& kaggle kernels status $Kernel 2>&1 | Out-String).Trim()
            $statusExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorAction
        }
        if ($statusExitCode -eq 0 -and $lastText -match 'KernelWorkerStatus\.([A-Z_]+)') {
            return $Matches[1]
        }
        if ($AllowAbsent -and $lastText -match "Cannot access kernel") {
            return "ABSENT"
        }
        if ($attempt -lt 3) {
            Start-Sleep -Seconds 2
        }
    }
    throw "Could not read kernel state for ${Kernel}: $lastText"
}

try {
    $hasMutex = $mutex.WaitOne(0)
    if (-not $hasMutex) {
        Write-Host "Another LOEO watcher already holds the global mutex; exiting."
        exit 0
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxHours)
    $contracts = @(
        @{
            Embryo = "44b6"
            Parent = "dmitriigluzdov/biohub-exp009-loeo-holdout-44b6"
            Audit = "dmitriigluzdov/biohub-exp011-audit-loeo-44b6"
        },
        @{
            Embryo = "6bba"
            Parent = "dmitriigluzdov/biohub-exp010-loeo-holdout-6bba"
            Audit = "dmitriigluzdov/biohub-exp012-audit-loeo-6bba"
        }
    )

    while ($true) {
        $allAuditsActive = $true
        foreach ($contract in $contracts) {
            $parentState = Get-KernelState -Kernel $contract.Parent
            $auditState = Get-KernelState -Kernel $contract.Audit -AllowAbsent
            $timestamp = [DateTimeOffset]::UtcNow.ToString("o")
            Write-Host "$timestamp embryo=$($contract.Embryo) parent=$parentState audit=$auditState"

            if ($parentState -in @("ERROR", "FAILED", "CANCELLED")) {
                throw "Parent $($contract.Parent) terminated in state $parentState"
            }
            if ($auditState -in @("ERROR", "FAILED", "CANCELLED")) {
                throw "Audit $($contract.Audit) terminated in state $auditState"
            }

            if ($auditState -in @("QUEUED", "RUNNING", "COMPLETE")) {
                continue
            }
            $allAuditsActive = $false
            if ($parentState -eq "COMPLETE") {
                & (Join-Path $PSScriptRoot "launch_loeo_audit.ps1") -Embryo $contract.Embryo
                if ($LASTEXITCODE -ne 0) {
                    throw "Audit launch failed for embryo $($contract.Embryo)"
                }
                $launchedState = Get-KernelState -Kernel $contract.Audit
                if ($launchedState -notin @("QUEUED", "RUNNING", "COMPLETE")) {
                    throw "Unexpected post-launch audit state $launchedState for $($contract.Embryo)"
                }
            }
        }

        if ($allAuditsActive) {
            Write-Host "All LOEO audits are active or complete; watcher finished."
            break
        }
        if ($Once) {
            Write-Host "Single status pass complete; no waiting requested."
            break
        }
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "LOEO watcher exceeded its ${MaxHours}h deadline"
        }
        Start-Sleep -Seconds $PollSeconds
    }
}
finally {
    if ($hasMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
