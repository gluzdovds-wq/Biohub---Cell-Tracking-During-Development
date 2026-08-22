param(
    [ValidateRange(30, 600)]
    [int]$PollSeconds = 60,
    [ValidateRange(1, 36)]
    [int]$MaxHours = 24,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$mutex = [System.Threading.Mutex]::new($false, "Global\BiohubExp039LaunchWatcher")
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
    $auditPath = Join-Path $repoRoot "outputs/exp038_kaggle_v3/exp038_checkpoint_audit.json"
    $buildPath = Join-Path $repoRoot "kaggle_notebooks/exp039_own_seed_secondary_ablation/build_receipt.json"
    $notebookPath = Join-Path $repoRoot "kaggle_notebooks/exp039_own_seed_secondary_ablation/own_seed_secondary_ablation.ipynb"

    if (-not (Test-Path -LiteralPath $auditPath -PathType Leaf)) {
        throw "Missing EXP038 v3 receipt: $auditPath"
    }
    $auditSha = (Get-FileHash -LiteralPath $auditPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($auditSha -ne "bb9b53865d5c1583a7942484c3c04e7e2de508b93f1a214553f6e0daad0ef9c4") {
        throw "EXP038 v3 receipt SHA drift: $auditSha"
    }
    $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    if ($audit.status -ne "PASS_NEW_COMPATIBLE_CHECKPOINT" -or
        $audit.loader_contract.status -ne "PASS_UNCHANGED_EXP006_LOADER_CONTRACT") {
        throw "EXP038 compatibility/loader contract did not pass"
    }

    $build = Get-Content -LiteralPath $buildPath -Raw | ConvertFrom-Json
    if ($build.status -ne "PASS_EXACT_SECONDARY_CHECKPOINT_ABLATION_BUILD") {
        throw "EXP039 build receipt did not pass"
    }
    $notebookSha = (Get-FileHash -LiteralPath $notebookPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($notebookSha -ne $build.output_sha256 -or
        $notebookSha -ne "2e42486f735d3eb8401bb8c8f3eb63b63f9454457fd086a68358afdb51d2bf7d") {
        throw "EXP039 notebook SHA drift: $notebookSha"
    }
}

try {
    $hasMutex = $mutex.WaitOne(0)
    if (-not $hasMutex) {
        Write-Host "Another EXP039 launch watcher already holds the global mutex; exiting."
        exit 0
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxHours)
    $audits = @(
        "dmitriigluzdov/biohub-exp011-audit-loeo-44b6",
        "dmitriigluzdov/biohub-exp012-audit-loeo-6bba"
    )
    $candidate = "dmitriigluzdov/biohub-exp039-own-seed-secondary-ablation"

    while ($true) {
        $candidateState = Get-KernelState -Kernel $candidate -AllowAbsent
        if ($candidateState -in @("QUEUED", "RUNNING", "COMPLETE")) {
            Write-Host "EXP039 is already active or complete ($candidateState); watcher finished."
            break
        }
        if ($candidateState -in @("ERROR", "FAILED", "CANCELLED")) {
            throw "EXP039 already terminated in state $candidateState"
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
        Write-Host "$timestamp audits=$($states -join ',') exp039=$candidateState"

        if (@($states | Where-Object { $_ -ne "COMPLETE" }).Count -eq 0) {
            Assert-LocalContracts
            $kernelDir = Join-Path (Split-Path $PSScriptRoot -Parent) "kaggle_notebooks/exp039_own_seed_secondary_ablation"
            & kaggle kernels push -p $kernelDir
            if ($LASTEXITCODE -ne 0) {
                throw "EXP039 kernel push failed"
            }
            $launchedState = Get-KernelState -Kernel $candidate
            if ($launchedState -notin @("QUEUED", "RUNNING", "COMPLETE")) {
                throw "Unexpected EXP039 post-launch state: $launchedState"
            }
            Write-Host "EXP039 launched after both untouched LOEO audits completed."
            break
        }

        if ($Once) {
            Write-Host "Single gate pass complete; EXP039 was not launched."
            break
        }
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "EXP039 launch watcher exceeded its ${MaxHours}h deadline"
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
