param(
    [ValidateRange(30, 600)]
    [int]$PollSeconds = 60,
    [ValidateRange(1, 24)]
    [int]$MaxHours = 18,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$competition = "biohub-cell-tracking-during-development"
$mutex = [System.Threading.Mutex]::new($false, "Global\BiohubChildSubmissionWatcher")
$hasMutex = $false

$messages = @{
    Baseline = "EXP-006 harmonic dual-seed guarded division frontier"
    Parent14 = "EXP-014 coordinate-only detector consensus"
    Parent19 = "EXP-019 intensity COM coordinate refinement"
    Child22 = "EXP-022 detector intensity coordinate blend"
    Child23 = "EXP-023 agreement gated coordinate blend"
}

function Invoke-KaggleRead([string[]]$CliArguments, [string]$Description) {
    $lastText = ""
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $lastText = (& kaggle @CliArguments 2>&1 | Out-String).Trim()
            $statusExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorAction
        }
        if ($statusExitCode -eq 0) {
            return $lastText
        }
        if ($attempt -lt 3) {
            Start-Sleep -Seconds 2
        }
    }
    throw "Could not read ${Description}: $lastText"
}

function Find-Submission([object[]]$Rows, [string]$Description) {
    return @(
        $Rows |
            Where-Object { $_.description -eq $Description } |
            Sort-Object { [DateTime]$_.date } -Descending
    ) | Select-Object -First 1
}

function Read-PublicScore([object]$Submission) {
    if ($null -eq $Submission -or
        $Submission.status -ne "SubmissionStatus.COMPLETE" -or
        [string]::IsNullOrWhiteSpace($Submission.publicScore)) {
        return $null
    }
    return [double]::Parse(
        $Submission.publicScore,
        [Globalization.CultureInfo]::InvariantCulture
    )
}

try {
    $hasMutex = $mutex.WaitOne(0)
    if (-not $hasMutex) {
        Write-Host "Another child-submission watcher already holds the global mutex; exiting."
        exit 0
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxHours)
    $queue = @(
        @{ Candidate = "EXP022"; Message = $messages.Child22 },
        @{ Candidate = "EXP023"; Message = $messages.Child23 }
    )

    while ($true) {
        $submissionText = Invoke-KaggleRead `
            @("competitions", "submissions", "-c", $competition, "-v") `
            "competition submissions"
        $rows = @($submissionText | ConvertFrom-Csv)

        $pendingChildren = @(
            $queue | Where-Object {
                $null -eq (Find-Submission $rows $_.Message)
            }
        )
        if (-not $pendingChildren.Count) {
            Write-Host "Both conditional child candidates are already registered; watcher finished."
            break
        }

        $baselineSubmission = Find-Submission $rows $messages.Baseline
        $parent14Submission = Find-Submission $rows $messages.Parent14
        $parent19Submission = Find-Submission $rows $messages.Parent19
        $baselineScore = Read-PublicScore $baselineSubmission
        $parent14Score = Read-PublicScore $parent14Submission
        $parent19Score = Read-PublicScore $parent19Submission
        $timestamp = [DateTimeOffset]::UtcNow.ToString("o")

        if ($null -eq $baselineScore -or $null -eq $parent14Score -or $null -eq $parent19Score) {
            $baselineState = if ($null -eq $baselineSubmission) { "ABSENT" } else { $baselineSubmission.status }
            $parent14State = if ($null -eq $parent14Submission) { "ABSENT" } else { $parent14Submission.status }
            $parent19State = if ($null -eq $parent19Submission) { "ABSENT" } else { $parent19Submission.status }
            Write-Host "$timestamp gate=WAIT baseline=$baselineState parent14=$parent14State parent19=$parent19State"
        }
        elseif ($parent14Score -lt $baselineScore -or $parent19Score -lt $baselineScore) {
            Write-Host "$timestamp gate=REJECT baseline=$baselineScore parent14=$parent14Score parent19=$parent19Score"
            Write-Host "At least one parent is worse than EXP006; no conditional child will be submitted."
            break
        }
        else {
            $limitText = Invoke-KaggleRead `
                @("competitions", "submission-limits", "-c", $competition) `
                "submission quota"
            if ($limitText -notmatch 'Remaining today:\s+(\d+)') {
                throw "Could not parse remaining submission quota: $limitText"
            }
            $remaining = [int]$Matches[1]
            Write-Host "$timestamp gate=PASS baseline=$baselineScore parent14=$parent14Score parent19=$parent19Score remaining=$remaining next=$($pendingChildren[0].Candidate)"

            if ($remaining -gt 0) {
                # The helper revalidates kernel completion, the immutable artifact
                # SHA, schema, graph invariants, and quota before one mutation.
                & (Join-Path $PSScriptRoot "submit_candidate.ps1") `
                    -Candidate $pendingChildren[0].Candidate `
                    -Submit
                if ($LASTEXITCODE -ne 0) {
                    throw "Submission helper failed for $($pendingChildren[0].Candidate)"
                }
                Start-Sleep -Seconds 30
                continue
            }
        }

        if ($Once) {
            Write-Host "Single child-gate pass complete; no submission was requested."
            break
        }
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "Child-submission watcher exceeded its ${MaxHours}h deadline"
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
