param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("44b6", "6bba")]
    [string]$Embryo,
    [string]$OutputRoot = "outputs\verified_loeo_parents"
)

$ErrorActionPreference = "Stop"

$contracts = @{
    "44b6" = @{
        Kernel = "dmitriigluzdov/biohub-exp009-loeo-holdout-44b6"
        Epochs = 5
        Train = 128
        Checkpoint = 4
        Calibration = 8
        Audit = 63
    }
    "6bba" = @{
        Kernel = "dmitriigluzdov/biohub-exp010-loeo-holdout-6bba"
        Epochs = 10
        Train = 71
        Checkpoint = 4
        Calibration = 8
        Audit = 120
    }
}
$expected = $contracts[$Embryo]
$statusText = (& kaggle kernels status $expected.Kernel 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not read parent status: $statusText"
}
if ($statusText -notmatch 'KernelWorkerStatus\.COMPLETE') {
    throw "Parent is not COMPLETE: $statusText"
}

$destination = Join-Path $OutputRoot $Embryo
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$contractName = "loeo_${Embryo}_contract.json"
$weightRelative = "loeo_holdout_${Embryo}/edge_predictor_best.pth"
$pattern = "($([regex]::Escape($contractName))|$([regex]::Escape($weightRelative)))$"
& kaggle kernels output $expected.Kernel -p $destination --file-pattern $pattern --page-size 20
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download the parent contract/checkpoint"
}

$contractPath = Join-Path $destination $contractName
$weightPath = Join-Path $destination $weightRelative
if (-not (Test-Path -LiteralPath $contractPath -PathType Leaf)) {
    throw "Missing downloaded contract: $contractPath"
}
if (-not (Test-Path -LiteralPath $weightPath -PathType Leaf)) {
    throw "Missing downloaded checkpoint: $weightPath"
}

$resolvedContractPath = (Resolve-Path -LiteralPath $contractPath).Path
$contractBytes = [System.IO.File]::ReadAllBytes($resolvedContractPath)
$contract = [System.Text.Encoding]::UTF8.GetString($contractBytes) | ConvertFrom-Json
foreach ($check in @(
    @("status", $contract.status, "training_complete"),
    @("holdout_embryo", $contract.holdout_embryo, $Embryo),
    @("seed", [int]$contract.seed, 314159),
    @("epochs", [int]$contract.epochs, [int]$expected.Epochs)
)) {
    if ($check[1] -ne $check[2]) {
        throw "Contract mismatch for $($check[0]): observed '$($check[1])', expected '$($check[2])'"
    }
}

$train = @($contract.train | ForEach-Object { [string]$_ })
$checkpoint = @($contract.checkpoint_validation | ForEach-Object { [string]$_ })
$calibration = @($contract.calibration | ForEach-Object { [string]$_ })
$audit = @($contract.audit | ForEach-Object { [string]$_ })
$observedSizes = @{
    Train = $train.Count
    Checkpoint = $checkpoint.Count
    Calibration = $calibration.Count
    Audit = $audit.Count
}
foreach ($name in $observedSizes.Keys) {
    if ($observedSizes[$name] -ne [int]$expected[$name]) {
        throw "Split-size mismatch for ${name}: $($observedSizes[$name]) vs $($expected[$name])"
    }
}

function Assert-Unique([string]$Name, [string[]]$Values) {
    if (@($Values | Sort-Object -Unique).Count -ne $Values.Count) {
        throw "Duplicate movie in $Name split"
    }
}
function Assert-Disjoint([string]$LeftName, [string[]]$Left, [string]$RightName, [string[]]$Right) {
    $overlap = @($Left | Where-Object { $Right -contains $_ } | Sort-Object -Unique)
    if ($overlap.Count) {
        throw "Split overlap ${LeftName}/${RightName}: $($overlap -join ', ')"
    }
}

Assert-Unique "train" $train
Assert-Unique "checkpoint_validation" $checkpoint
Assert-Unique "calibration" $calibration
Assert-Unique "audit" $audit
Assert-Disjoint "train" $train "calibration" $calibration
Assert-Disjoint "train" $train "audit" $audit
Assert-Disjoint "calibration" $calibration "audit" $audit
foreach ($name in $checkpoint) {
    if ($calibration -notcontains $name) {
        throw "Checkpoint-validation movie is absent from calibration: $name"
    }
}

foreach ($name in $train) {
    if ($name.StartsWith("${Embryo}_")) {
        throw "Held-out embryo leaked into train: $name"
    }
}
foreach ($name in @($calibration + $audit)) {
    if (-not $name.StartsWith("${Embryo}_")) {
        throw "Non-held-out embryo found outside train: $name"
    }
}

$weightItem = Get-Item -LiteralPath $weightPath
$weightSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $weightPath).Hash.ToLowerInvariant()
$artifactProperty = $contract.artifacts.PSObject.Properties["edge_predictor_best.pth"]
if ($null -eq $artifactProperty) {
    throw "Missing edge_predictor_best.pth artifact receipt"
}
$artifactReceipt = $artifactProperty.Value
if ([int64]$artifactReceipt.bytes -ne $weightItem.Length) {
    throw "Checkpoint byte mismatch: $($weightItem.Length) vs $($artifactReceipt.bytes)"
}
if ([string]$artifactReceipt.sha256 -ne $weightSha) {
    throw "Checkpoint SHA mismatch: $weightSha vs $($artifactReceipt.sha256)"
}
$contractSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $contractPath).Hash.ToLowerInvariant()

[pscustomobject]@{
    status = "PASS"
    embryo = $Embryo
    kernel = $expected.Kernel
    epochs = [int]$expected.Epochs
    train = $train.Count
    checkpoint_validation = $checkpoint.Count
    calibration = $calibration.Count
    audit = $audit.Count
    contract_path = $resolvedContractPath
    contract_sha256 = $contractSha
    checkpoint_path = (Resolve-Path -LiteralPath $weightPath).Path
    checkpoint_bytes = $weightItem.Length
    checkpoint_sha256 = $weightSha
} | Format-List
