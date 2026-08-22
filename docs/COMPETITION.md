# Competition specification

Verified 2026-08-22 from the Kaggle competition page and the official `royerlab/kaggle-cell-tracking-competition` repository.

## Task

For each 3D microscopy movie of a developing zebrafish embryo:

1. detect all cell centres in every frame;
2. link the same cell across adjacent frames;
3. detect mitoses and link one parent to two daughters;
4. output the resulting lineage graph as `submission.csv`.

## Data

- Image: OME-Zarr `uint16`, logical axes `(T, Z, Y, X)`.
- Physical voxel scale: `Z=1.625`, `Y=X=0.40625` micrometres/voxel; Z is four times coarser.
- Train: 199 cropped movies from two embryos (`44b6`: 71, `6bba`: 128), each with `.zarr` images and sparse `.geff` graph labels.
- Visible test: four example movies; Kaggle replaces them with the hidden set during code rerun.
- GEFF nodes contain `(t,z,y,x)` centroids; directed edges connect temporal continuations; a division is a node with two outgoing edges.
- Labels are sparse. An unlabelled cell is unknown, not a negative example.
- Competition inventory reported by the Kaggle data page: about 87.61 GB and 24,886 files. Public/private hidden-test split is 29%/71%.

The train crops are not independent biological units. Validation must hold out an embryo, not randomly split crops.

## Metric

```text
score = adjusted_edge_jaccard + 0.1 * division_jaccard
```

Node matching is per timepoint by optimal bipartite assignment in physical coordinates, with maximum distance 7.0 micrometres. A predicted edge is correct only when both endpoints match ground-truth nodes and the corresponding ground-truth edge exists.

```text
edge_jaccard = TP / (TP + FP + FN)
```

The edge score is adjusted for over-predicting the estimated total number of nodes. Under-prediction is not penalised by that adjustment, although it naturally creates edge false negatives. Per-sample edge scores are weighted by `TP+FP+FN`; division counts are micro-averaged across samples. Scores can exceed 1.0 because of the node-count adjustment.

Division matching is topology-aware: a predicted connected component must cover the pre-split region and touch both daughter lineages. Because division Jaccard has weight 0.1, association quality dominates.

## Submission contract

Columns:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

- Node row: valid `node_id,t,z,y,x`; edge IDs are `-1`.
- Edge row: valid `source_id,target_id`; node/coordinate fields are `-1`.
- `id` is consecutive.
- `dataset` is the test folder stem.
- Every hidden dataset must appear.
- Output filename must be exactly `submission.csv`.

## Code constraints

- Submission through Kaggle Notebooks only.
- CPU or GPU runtime at most 12 hours.
- Internet disabled.
- Public external data and pretrained weights are allowed.
- Five submissions per day.
- Entry/team deadline: 2026-09-22 23:59 UTC; final deadline: 2026-09-29 23:59 UTC.

## Primary sources

- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview
- https://github.com/royerlab/kaggle-cell-tracking-competition
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
