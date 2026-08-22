param(
    [ValidateRange(30, 600)]
    [int]$PollSeconds = 45,
    [ValidateRange(1, 12)]
    [int]$MaxHours = 4,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$kernel = "dmitriigluzdov/biohub-exp053-coordinate-registered"
$expectedSha = "8103351bf371b7a0654ae87a384e82862a75d33ed83759500d7507c40ee802bc"
$outputDir = Join-Path $repoRoot "outputs\exp053_kaggle_v1"
$artifact = Join-Path $outputDir "submission.csv"
$receiptPath = Join-Path $outputDir "exp053_receipt.json"
$stateDir = Join-Path $repoRoot "outputs\exp053_proxy_watcher"
$marker = Join-Path $stateDir "exp034_v11_pushed.marker"
$proxySource = Join-Path $repoRoot "kaggle_notebooks\exp034_coordinate_proxy_audit\coordinate_proxy_audit.py"
$proxyMetadata = Join-Path $repoRoot "kaggle_notebooks\exp034_coordinate_proxy_audit\kernel-metadata.json"
$expectedProxySourceSha = "ef09c2638b96df62553202b85db21ebd2ecab2801f3d5f339f810ae102f7c7f5"
$expectedProxyMetadataSha = "43ff6d8ae68730c6dbb20099e50b9bd8a3fd1ba3094f8c49cf55873472cff4a1"
$mutex = [System.Threading.Mutex]::new($false, "Global\BiohubExp053ProxyWatcher")
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
    throw "Could not read EXP053 kernel state: $lastText"
}

try {
    $hasMutex = $mutex.WaitOne(0)
    if (-not $hasMutex) {
        Write-Host "Another EXP053 proxy watcher holds the mutex; exiting."
        exit 0
    }
    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxHours)
    while ($true) {
        $state = Get-KernelState $kernel
        $timestamp = [DateTimeOffset]::UtcNow.ToString("o")
        Write-Host "$timestamp exp053=$state"
        if ($state -in @("ERROR", "FAILED", "CANCELLED")) {
            throw "EXP053 reached terminal failure state $state"
        }
        if ($state -eq "COMPLETE") {
            if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
                New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
                & kaggle kernels output $kernel -p $outputDir
                if ($LASTEXITCODE -ne 0) {
                    throw "Could not download EXP053 output"
                }
            }
            $observedSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact).Hash.ToLowerInvariant()
            if ($observedSha -ne $expectedSha) {
                throw "EXP053 Kaggle SHA mismatch: $observedSha != $expectedSha"
            }
            $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
            if ($receipt.status -ne "PASS_IMMUTABLE_EXP053_COMPOSITION" -or
                $receipt.output_sha256 -ne $expectedSha -or
                $receipt.node_rows_exact_from_exp014 -ne $true -or
                $receipt.edge_rows_exact_from_exp052 -ne $true -or
                $receipt.submission_allowed_by_this_receipt -ne $false) {
                throw "EXP053 receipt contract failed"
            }
            & (Join-Path $PSScriptRoot "audit_submission.ps1") `
                -Path $artifact `
                -ExpectedDatasetCount 4
            if ($LASTEXITCODE -ne 0) {
                throw "EXP053 full submission audit failed"
            }

            if (Test-Path -LiteralPath $marker -PathType Leaf) {
                Write-Host "EXP034 v11 launch marker already exists; watcher finished."
                break
            }
            $sourceSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $proxySource).Hash.ToLowerInvariant()
            $metadataSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $proxyMetadata).Hash.ToLowerInvariant()
            if ($sourceSha -ne $expectedProxySourceSha -or $metadataSha -ne $expectedProxyMetadataSha) {
                throw "EXP034 v11 source contract drift: source=$sourceSha metadata=$metadataSha"
            }
            & kaggle kernels push -p (Split-Path -Parent $proxySource)
            if ($LASTEXITCODE -ne 0) {
                throw "Could not push EXP034 overlap audit v11"
            }
            New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
            Set-Content -LiteralPath $marker -Value "$timestamp sha256=$observedSha" -Encoding utf8
            Write-Host "PASS: EXP053 independently audited and reject-only EXP034 v11 pushed."
            break
        }

        if ($Once) {
            Write-Host "Single EXP053 status pass complete; no mutation requested."
            break
        }
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "EXP053 watcher exceeded its ${MaxHours}h deadline"
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
