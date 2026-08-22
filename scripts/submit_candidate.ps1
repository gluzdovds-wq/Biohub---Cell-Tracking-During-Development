param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("EXP014", "EXP019", "EXP022", "EXP023")]
    [string]$Candidate,
    [switch]$Submit
)

$ErrorActionPreference = "Stop"
$competition = "biohub-cell-tracking-during-development"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-KaggleRead([string[]]$CliArguments, [string]$Description) {
    $lastText = ""
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $lastText = (& kaggle @CliArguments 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -eq 0) {
            return $lastText
        }
        if ($attempt -lt 3) {
            Start-Sleep -Seconds 2
        }
    }
    throw "Could not read ${Description} after 3 attempts: $lastText"
}

$candidates = @{
    "EXP014" = @{
        Kernel = "dmitriigluzdov/biohub-exp014-coordinate-ensemble"
        Path = "outputs\exp014_wrapper_v2\submission.csv"
        Sha256 = "c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0"
        Message = "EXP-014 coordinate-only detector consensus"
    }
    "EXP019" = @{
        Kernel = "dmitriigluzdov/biohub-exp019-intensity-coordinate-refine"
        Path = "outputs\exp019_intensity_coordinate_refine\submission.csv"
        Sha256 = "7487ecb7de8c110caffd35bd043902b484ee4634ec58d020caebabfad9296c6d"
        Message = "EXP-019 intensity COM coordinate refinement"
    }
    "EXP022" = @{
        Kernel = "dmitriigluzdov/biohub-exp022-coordinate-blend"
        Path = "outputs\exp022_kaggle_v1\submission.csv"
        Sha256 = "91e24e750dc2a305943713618bbaa3f0de95283cbeb2de9e9b2d6ecef3f8fb6a"
        Message = "EXP-022 detector intensity coordinate blend"
    }
    "EXP023" = @{
        Kernel = "dmitriigluzdov/biohub-exp023-agreement-gated-coordinates"
        Path = "outputs\exp023_kaggle_v1\submission.csv"
        Sha256 = "8bff01ab65cc2f9e022684822cd09240265417567abd5406387b808f7e052de3"
        Message = "EXP-023 agreement gated coordinate blend"
    }
}
$selected = $candidates[$Candidate]
$artifactPath = Join-Path $repoRoot $selected.Path

$kernelStatus = Invoke-KaggleRead @("kernels", "status", $selected.Kernel) "candidate-kernel status"
if ($kernelStatus -notmatch 'KernelWorkerStatus\.COMPLETE') {
    throw "Candidate kernel is not COMPLETE: $kernelStatus"
}

if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
    throw "Canonical local artifact is missing: $artifactPath"
}
$observedSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifactPath).Hash.ToLowerInvariant()
if ($observedSha -ne $selected.Sha256) {
    throw "Artifact SHA mismatch: $observedSha vs $($selected.Sha256)"
}

& (Join-Path $PSScriptRoot "audit_submission.ps1") -Path $artifactPath -ExpectedDatasetCount 4

$limitText = Invoke-KaggleRead @("competitions", "submission-limits", "-c", $competition) "submission quota"
if ($limitText -notmatch 'Remaining today:\s+(\d+)') {
    throw "Could not parse remaining submission quota: $limitText"
}
$remaining = [int]$Matches[1]

Write-Host "Candidate validation: PASS"
Write-Host "candidate=$Candidate"
Write-Host "kernel=$($selected.Kernel)"
Write-Host "sha256=$observedSha"
Write-Host "remaining_today=$remaining"
if (-not $Submit) {
    Write-Host "Dry run only; pass -Submit to spend one competition submission."
    exit 0
}
if ($remaining -lt 1) {
    throw "No competition submission slots remain today"
}

& kaggle competitions submit $competition -k $selected.Kernel -f submission.csv -m $selected.Message
if ($LASTEXITCODE -ne 0) {
    throw "Submission command failed for $Candidate"
}
Write-Host "Submission requested: $Candidate"
