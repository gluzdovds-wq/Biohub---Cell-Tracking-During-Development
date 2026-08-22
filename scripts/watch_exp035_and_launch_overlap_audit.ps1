param(
    [ValidateRange(30, 600)]
    [int]$PollSeconds = 60,
    [ValidateRange(1, 6)]
    [int]$MaxHours = 3,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$kernel = "dmitriigluzdov/biohub-exp035-guarded-division-consensus"
$expectedSha = "7376bd3c4056ee7c7f82fadd2db3bb37230ad09e399ae1f815c3c53a51374bd4"
$outputDir = Join-Path $repoRoot "outputs\exp035_kaggle_v2"
$artifact = Join-Path $outputDir "submission.csv"
$stateDir = Join-Path $repoRoot "outputs\exp035_audit_watcher"
$marker = Join-Path $stateDir "exp034_v4_pushed.marker"
$mutex = [System.Threading.Mutex]::new($false, "Global\BiohubExp035AuditWatcher")
$hasMutex = $false

function Get-KernelState([string]$Kernel) {
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
        if ($attempt -lt 3) {
            Start-Sleep -Seconds 2
        }
    }
    throw "Could not read EXP035 kernel state: $lastText"
}

try {
    $hasMutex = $mutex.WaitOne(0)
    if (-not $hasMutex) {
        Write-Host "Another EXP035 audit watcher holds the global mutex; exiting."
        exit 0
    }
    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxHours)
    while ($true) {
        $state = Get-KernelState $kernel
        $timestamp = [DateTimeOffset]::UtcNow.ToString("o")
        Write-Host "$timestamp kernel_state=$state"
        if ($state -eq "ERROR" -or $state -eq "CANCELLED") {
            throw "EXP035 reached terminal failure state $state"
        }
        if ($state -eq "COMPLETE") {
            if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
                New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
                & kaggle kernels output $kernel -p $outputDir
                if ($LASTEXITCODE -ne 0) {
                    throw "Could not download EXP035 output"
                }
            }
            $observedSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact).Hash.ToLowerInvariant()
            if ($observedSha -ne $expectedSha) {
                throw "EXP035 Kaggle SHA mismatch: $observedSha != $expectedSha"
            }
            & (Join-Path $PSScriptRoot "audit_submission.ps1") `
                -Path $artifact `
                -ExpectedDatasetCount 4

            if (Test-Path -LiteralPath $marker -PathType Leaf) {
                Write-Host "EXP034 v4 launch marker already exists; watcher finished."
                break
            }
            & kaggle kernels push -p (Join-Path $repoRoot "kaggle_notebooks\exp034_coordinate_proxy_audit")
            if ($LASTEXITCODE -ne 0) {
                throw "Could not push EXP034 overlap audit v4"
            }
            New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
            Set-Content -LiteralPath $marker -Value "$timestamp sha256=$observedSha" -Encoding utf8
            Write-Host "PASS: EXP035 independently audited and EXP034 v4 pushed."
            break
        }

        if ($Once) {
            Write-Host "Single EXP035 status pass complete; no mutation requested."
            break
        }
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "EXP035 watcher exceeded its ${MaxHours}h deadline"
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
