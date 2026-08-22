param(
    [ValidateRange(30, 600)]
    [int]$PollSeconds = 45,
    [ValidateRange(1, 36)]
    [int]$MaxHours = 24,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$mutex = [System.Threading.Mutex]::new($false, "Global\BiohubExp049LaunchWatcher")
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

function Assert-LocalContracts {
    $repoRoot = Split-Path $PSScriptRoot -Parent
    $sourcePath = Join-Path $repoRoot "kaggle_notebooks/exp049_cross_embryo_production/cross_embryo_production.py"
    $metadataPath = Join-Path $repoRoot "kaggle_notebooks/exp049_cross_embryo_production/kernel-metadata.json"
    $sourceSha = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceSha -ne "fb164ff9da63399a6683aca856e69044c25a1a68e2998fcf5dd7520f9067fc2e") {
        throw "EXP049 source SHA drift: $sourceSha"
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($metadata.id -ne "dmitriigluzdov/biohub-exp049-cross-embryo-production" -or
        $metadata.code_file -ne "cross_embryo_production.py" -or
        @($metadata.kernel_sources).Count -ne 4) {
        throw "EXP049 metadata contract drift"
    }
}

try {
    $hasMutex = $mutex.WaitOne(0)
    if (-not $hasMutex) {
        Write-Host "Another EXP049 launch watcher holds the mutex; exiting."
        exit 0
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxHours)
    $audits = @(
        "dmitriigluzdov/biohub-exp011-audit-loeo-44b6",
        "dmitriigluzdov/biohub-exp012-audit-loeo-6bba"
    )
    $candidate = "dmitriigluzdov/biohub-exp049-cross-embryo-production"

    while ($true) {
        $candidateState = Get-KernelState -Kernel $candidate -AllowAbsent
        if ($candidateState -in @("QUEUED", "RUNNING", "COMPLETE")) {
            Write-Host "EXP049 is already active or complete ($candidateState); watcher finished."
            break
        }
        if ($candidateState -in @("ERROR", "FAILED", "CANCELLED")) {
            throw "EXP049 already terminated in state $candidateState"
        }

        $states = @()
        foreach ($audit in $audits) {
            $state = Get-KernelState -Kernel $audit -AllowAbsent
            if ($state -in @("ERROR", "FAILED", "CANCELLED")) {
                throw "Required LOEO audit $audit terminated in state $state"
            }
            $states += $state
        }
        $timestamp = [DateTimeOffset]::UtcNow.ToString("o")
        Write-Host "$timestamp audits=$($states -join ',') exp049=$candidateState"

        if (@($states | Where-Object { $_ -ne "COMPLETE" }).Count -eq 0) {
            Assert-LocalContracts
            $kernelDir = Join-Path (Split-Path $PSScriptRoot -Parent) "kaggle_notebooks/exp049_cross_embryo_production"
            & kaggle kernels push -p $kernelDir
            if ($LASTEXITCODE -ne 0) {
                throw "EXP049 kernel push failed"
            }
            $launchedState = Get-KernelState -Kernel $candidate
            if ($launchedState -notin @("QUEUED", "RUNNING", "COMPLETE")) {
                throw "Unexpected EXP049 post-launch state: $launchedState"
            }
            Write-Host "EXP049 launched after both untouched LOEO audits completed."
            break
        }

        if ($Once) {
            Write-Host "Single gate pass complete; EXP049 was not launched."
            break
        }
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "EXP049 launch watcher exceeded its ${MaxHours}h deadline"
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
