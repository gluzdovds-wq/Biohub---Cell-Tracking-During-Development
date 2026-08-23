param(
    [ValidateRange(15, 600)]
    [int]$PollSeconds = 60,
    [ValidateRange(1, 24)]
    [int]$MaxHours = 16,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$mutex = [System.Threading.Mutex]::new($false, "Global\BiohubWeakTiebreakWatcher")
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
        Write-Host "Another weak tie-break watcher holds the mutex; exiting."
        exit 0
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxHours)
    $contracts = @(
        @{
            Embryo = "44b6"
            Upstream = "dmitriigluzdov/biohub-exp011-audit-loeo-44b6"
            Target = "dmitriigluzdov/biohub-exp050-weak-tie-break-44b6"
        },
        @{
            Embryo = "6bba"
            Upstream = "dmitriigluzdov/biohub-exp012-audit-loeo-6bba"
            Target = "dmitriigluzdov/biohub-exp051-weak-tie-break-6bba"
        }
    )

    while ($true) {
        $allTargetsActive = $true
        foreach ($contract in $contracts) {
            $upstreamState = Get-KernelState -Kernel $contract.Upstream -AllowAbsent
            $targetState = Get-KernelState -Kernel $contract.Target -AllowAbsent
            $timestamp = [DateTimeOffset]::UtcNow.ToString("o")
            Write-Host "$timestamp embryo=$($contract.Embryo) upstream=$upstreamState weak_tiebreak=$targetState"

            if ($upstreamState -in @("ERROR", "FAILED", "CANCELLED")) {
                throw "Upstream $($contract.Upstream) terminated in state $upstreamState"
            }
            if ($targetState -in @("ERROR", "FAILED", "CANCELLED")) {
                throw "Weak tie-break audit $($contract.Target) terminated in state $targetState"
            }
            if ($targetState -in @("QUEUED", "RUNNING", "COMPLETE")) {
                continue
            }

            $allTargetsActive = $false
            if ($upstreamState -eq "COMPLETE") {
                & (Join-Path $PSScriptRoot "launch_weak_tiebreak_audit.ps1") -Embryo $contract.Embryo
                if ($LASTEXITCODE -ne 0) {
                    throw "Weak tie-break launch failed for embryo $($contract.Embryo)"
                }
                $launchedState = Get-KernelState -Kernel $contract.Target
                if ($launchedState -notin @("QUEUED", "RUNNING", "COMPLETE")) {
                    throw "Unexpected post-launch state $launchedState for $($contract.Embryo)"
                }
            }
        }

        if ($allTargetsActive) {
            Write-Host "Both weak tie-break audits are active or complete; watcher finished."
            break
        }
        if ($Once) {
            Write-Host "Single status pass complete."
            break
        }
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "Weak tie-break watcher exceeded its ${MaxHours}h deadline"
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
