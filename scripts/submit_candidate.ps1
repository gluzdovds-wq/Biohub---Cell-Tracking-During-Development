param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("EXP014", "EXP019", "EXP022", "EXP023", "EXP040", "EXP041", "EXP045", "EXP047", "EXP052", "EXP053", "EXP054", "EXP055")]
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
        Version = 2
        Path = "outputs\exp014_wrapper_v2\submission.csv"
        File = "submission.csv"
        Sha256 = "c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0"
        Message = "EXP-014 coordinate-only detector consensus"
        SubmissionReady = $false
    }
    "EXP019" = @{
        Kernel = "dmitriigluzdov/biohub-exp019-intensity-coordinate-refine"
        Version = 1
        Path = "outputs\exp019_intensity_coordinate_refine\submission.csv"
        File = "submission.csv"
        Sha256 = "7487ecb7de8c110caffd35bd043902b484ee4634ec58d020caebabfad9296c6d"
        Message = "EXP-019 intensity COM coordinate refinement"
        SubmissionReady = $false
    }
    "EXP022" = @{
        Kernel = "dmitriigluzdov/biohub-exp022-coordinate-blend"
        Version = 1
        Path = "outputs\exp022_kaggle_v1\submission.csv"
        File = "submission.csv"
        Sha256 = "91e24e750dc2a305943713618bbaa3f0de95283cbeb2de9e9b2d6ecef3f8fb6a"
        Message = "EXP-022 detector intensity coordinate blend"
        SubmissionReady = $false
    }
    "EXP023" = @{
        Kernel = "dmitriigluzdov/biohub-exp023-agreement-gated-coordinates"
        Version = 1
        Path = "outputs\exp023_kaggle_v1\submission.csv"
        File = "submission.csv"
        Sha256 = "8bff01ab65cc2f9e022684822cd09240265417567abd5406387b808f7e052de3"
        Message = "EXP-023 agreement gated coordinate blend"
        SubmissionReady = $false
    }
    "EXP040" = @{
        Kernel = "dmitriigluzdov/biohub-exp044-division-prune-artifacts"
        Version = 2
        Path = "outputs\exp044_kaggle_v2\exp040_submission.csv"
        File = "exp040_submission.csv"
        Sha256 = "9f0b0711b5ac0b078c5fb24332c2604c09118013116bc6fbe4d6f4e2eaa4a5e3"
        Message = "EXP-040 donor-consensus physical division prune"
        SubmissionReady = $false
    }
    "EXP041" = @{
        Kernel = "dmitriigluzdov/biohub-exp044-division-prune-artifacts"
        Version = 2
        Path = "outputs\exp044_kaggle_v2\exp041_submission.csv"
        File = "exp041_submission.csv"
        Sha256 = "21a42ffa33c8af7ef44b28f7edaea6a3d9666745139c9c51e132fed41a8fe114"
        Message = "EXP-041 strict donor-consensus physical division prune"
        SubmissionReady = $false
    }
    "EXP045" = @{
        Kernel = "dmitriigluzdov/biohub-exp046-coordinate-division-artifact"
        Version = 1
        Path = "outputs\exp046_kaggle_v1\exp045_submission.csv"
        File = "exp045_submission.csv"
        Sha256 = "4d93515ed72e76ea5be0d84c7a20d1e268e20ba37a8e4ce1ff50459d21399f88"
        Message = "EXP-045 coordinate consensus plus physical division prune"
        SubmissionReady = $false
    }
    "EXP047" = @{
        Kernel = "dmitriigluzdov/biohub-exp048-strict-coordinate-division"
        Version = 1
        Path = "outputs\exp048_kaggle_v1\exp047_submission.csv"
        File = "exp047_submission.csv"
        Sha256 = "5dd662d8d12f91120425a11a7667059529ce53ad7eab4f756879e9477cf363f2"
        Message = "EXP-047 coordinate consensus plus strict physical division prune"
        SubmissionReady = $false
    }
    "EXP052" = @{
        Kernel = "dmitriigluzdov/biohub-exp052-registered-relink"
        Version = 1
        Path = "outputs\exp052_kaggle_v1\submission.csv"
        File = "submission.csv"
        Sha256 = "3791f74f9247be99d3a9e673cd2ff9fd942764f1ad0b1d0a597d150b7a7c9fab"
        Message = "EXP-052 registered motion relink of EXP006 nodes"
        SubmissionReady = $false
    }
    "EXP053" = @{
        Kernel = "dmitriigluzdov/biohub-exp053-coordinate-registered"
        Version = 3
        Path = "outputs\exp053_kaggle_v3\submission.csv"
        File = "submission.csv"
        Sha256 = "9fd723827c65a5ad736b045a13a072da384de02e1ac2b1c57c8414335a38e6d5"
        Message = "EXP-053 coordinate consensus plus registered relink"
        SubmissionReady = $false
    }
    "EXP054" = @{
        Kernel = "dmitriigluzdov/biohub-exp054-registered-production"
        Version = 1
        Path = "outputs\exp054_kaggle_v1\submission.csv"
        File = "submission.csv"
        Sha256 = "09d692a4d00975ff474bcc63ee249accb21332ed21442b9fe9f46d7f20baf7a6"
        Message = "EXP-054 hidden-compatible registered motion relink"
        SubmissionReady = $false
    }
    "EXP055" = @{
        Kernel = "dmitriigluzdov/biohub-exp055-intensity-registered-production"
        Version = 1
        Path = "outputs\exp055_kaggle_v1\submission.csv"
        File = "submission.csv"
        Sha256 = "2db755ce91647915ced8c1bd4c65873b9e185c978c8807f1480613f1ba1d8fd1"
        Message = "EXP-055 hidden-compatible intensity coordinates plus registered relink"
        SubmissionReady = $false
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
Write-Host "kernel_version=$($selected.Version)"
Write-Host "sha256=$observedSha"
Write-Host "remaining_today=$remaining"
Write-Host "submission_ready=$($selected.SubmissionReady)"
if (-not $Submit) {
    Write-Host "Dry run only; pass -Submit to spend one competition submission."
    exit 0
}
if (-not $selected.SubmissionReady) {
    throw "$Candidate remains promotion-gated; immutable artifact validation alone cannot authorize submission"
}
if ($remaining -lt 1) {
    throw "No competition submission slots remain today"
}

& kaggle competitions submit $competition -k $selected.Kernel -v $selected.Version -f $selected.File -m $selected.Message
if ($LASTEXITCODE -ne 0) {
    throw "Submission command failed for $Candidate"
}
Write-Host "Submission requested: $Candidate"
