param(
    [ValidateRange(30, 600)]
    [int]$PollSeconds = 60,
    [ValidateRange(1, 18)]
    [int]$MaxHours = 12,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$competition = "biohub-cell-tracking-during-development"
$mutex = [System.Threading.Mutex]::new($false, "Global\BiohubSubmissionWatcher")
$hasMutex = $false

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

try {
    $hasMutex = $mutex.WaitOne(0)
    if (-not $hasMutex) {
        Write-Host "Another submission watcher already holds the global mutex; exiting."
        exit 0
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxHours)
    $queue = @(
        @{
            Candidate = "EXP014"
            Message = "EXP-014 coordinate-only detector consensus"
        },
        @{
            Candidate = "EXP019"
            Message = "EXP-019 intensity COM coordinate refinement"
        }
    )

    while ($true) {
        $submissionText = Invoke-KaggleRead `
            @("competitions", "submissions", "-c", $competition, "-v") `
            "competition submissions"
        $pending = @(
            $queue | Where-Object {
                $submissionText -notmatch [regex]::Escape($_.Message)
            }
        )
        if (-not $pending.Count) {
            Write-Host "All queued candidates are already registered; watcher finished."
            break
        }

        $limitText = Invoke-KaggleRead `
            @("competitions", "submission-limits", "-c", $competition) `
            "submission quota"
        if ($limitText -notmatch 'Remaining today:\s+(\d+)') {
            throw "Could not parse remaining submission quota: $limitText"
        }
        $remaining = [int]$Matches[1]
        $timestamp = [DateTimeOffset]::UtcNow.ToString("o")
        Write-Host "$timestamp remaining=$remaining next=$($pending[0].Candidate)"

        if ($remaining -gt 0) {
            # submit_candidate.ps1 repeats every immutable artifact/status/schema/SHA
            # check and issues exactly one non-retried mutating request.
            & (Join-Path $PSScriptRoot "submit_candidate.ps1") `
                -Candidate $pending[0].Candidate `
                -Submit
            if ($LASTEXITCODE -ne 0) {
                throw "Submission helper failed for $($pending[0].Candidate)"
            }
            Start-Sleep -Seconds 20
            continue
        }

        if ($Once) {
            Write-Host "Single quota pass complete; no slot was available."
            break
        }
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "Submission watcher exceeded its ${MaxHours}h deadline"
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
