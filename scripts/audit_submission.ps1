param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [int]$ExpectedDatasetCount = 0
)

$ErrorActionPreference = "Stop"

$expectedColumns = @(
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
)

if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Submission file not found: $Path"
}

$rows = @(Import-Csv -LiteralPath $Path)
if ($rows.Count -eq 0) {
    throw "Submission has no data rows: $Path"
}

$actualColumns = @($rows[0].PSObject.Properties.Name)
if (Compare-Object $actualColumns $expectedColumns -SyncWindow 0) {
    throw "Unexpected columns: $($actualColumns -join ',')"
}

$datasets = @($rows.dataset | Sort-Object -Unique)
if ($ExpectedDatasetCount -gt 0 -and $datasets.Count -ne $ExpectedDatasetCount) {
    throw "Expected $ExpectedDatasetCount datasets, found $($datasets.Count): $($datasets -join ',')"
}

$nodeLookup = @{}
$edgeKeys = [System.Collections.Generic.HashSet[string]]::new()
$inDegree = @{}
$outDegree = @{}
$nodeCount = 0
$edgeCount = 0

for ($index = 0; $index -lt $rows.Count; $index++) {
    $row = $rows[$index]
    if ([int64]$row.id -ne $index) {
        throw "Non-consecutive id at row ${index}: $($row.id)"
    }

    if ($row.row_type -eq "node") {
        $nodeCount++
        $nodeId = [int64]$row.node_id
        $key = "$($row.dataset)|$nodeId"
        if ($nodeLookup.ContainsKey($key)) {
            throw "Duplicate node_id within dataset: $key"
        }
        foreach ($field in @("t", "z", "y", "x")) {
            if ([int64]$row.$field -lt 0) {
                throw "Negative node field $field for $key"
            }
        }
        if ([int64]$row.source_id -ne -1 -or [int64]$row.target_id -ne -1) {
            throw "Node row has non-sentinel edge ids: $key"
        }
        $nodeLookup[$key] = [int64]$row.t
        continue
    }

    if ($row.row_type -eq "edge") {
        $edgeCount++
        foreach ($field in @("node_id", "t", "z", "y", "x")) {
            if ([int64]$row.$field -ne -1) {
                throw "Edge row has non-sentinel $field at id $($row.id)"
            }
        }
        $sourceId = [int64]$row.source_id
        $targetId = [int64]$row.target_id
        if ($sourceId -eq $targetId) {
            throw "Self-loop at id $($row.id)"
        }
        $edgeKey = "$($row.dataset)|$sourceId|$targetId"
        if (-not $edgeKeys.Add($edgeKey)) {
            throw "Duplicate edge: $edgeKey"
        }
        $sourceKey = "$($row.dataset)|$sourceId"
        $targetKey = "$($row.dataset)|$targetId"
        $outDegree[$sourceKey] = 1 + [int]($outDegree[$sourceKey] ?? 0)
        $inDegree[$targetKey] = 1 + [int]($inDegree[$targetKey] ?? 0)
        continue
    }

    throw "Unexpected row_type '$($row.row_type)' at id $($row.id)"
}

$nonConsecutiveEdges = 0
foreach ($edgeKey in $edgeKeys) {
    $parts = $edgeKey.Split('|')
    $dataset = $parts[0]
    $sourceKey = "$dataset|$($parts[1])"
    $targetKey = "$dataset|$($parts[2])"
    if (-not $nodeLookup.ContainsKey($sourceKey) -or -not $nodeLookup.ContainsKey($targetKey)) {
        throw "Orphan edge endpoint: $edgeKey"
    }
    if ($nodeLookup[$targetKey] -ne $nodeLookup[$sourceKey] + 1) {
        $nonConsecutiveEdges++
    }
}

if ($nonConsecutiveEdges -gt 0) {
    throw "Found $nonConsecutiveEdges edges that do not advance exactly one frame"
}

$maxInDegree = if ($inDegree.Count) { ($inDegree.Values | Measure-Object -Maximum).Maximum } else { 0 }
$maxOutDegree = if ($outDegree.Count) { ($outDegree.Values | Measure-Object -Maximum).Maximum } else { 0 }
if ($maxInDegree -gt 1) {
    throw "Maximum in-degree exceeds 1: $maxInDegree"
}
if ($maxOutDegree -gt 2) {
    throw "Maximum out-degree exceeds 2: $maxOutDegree"
}

[pscustomobject]@{
    path = (Resolve-Path -LiteralPath $Path).Path
    bytes = (Get-Item -LiteralPath $Path).Length
    rows = $rows.Count
    datasets = $datasets.Count
    dataset_names = $datasets -join "|"
    nodes = $nodeCount
    edges = $edgeCount
    edge_node_ratio = [math]::Round($edgeCount / [math]::Max(1, $nodeCount), 6)
    max_in_degree = $maxInDegree
    max_out_degree = $maxOutDegree
    status = "PASS"
} | Format-List
