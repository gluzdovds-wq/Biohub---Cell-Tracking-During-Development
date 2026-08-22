$ErrorActionPreference = "Stop"

$competition = "biohub-cell-tracking-during-development"
$kernels = @(
    @{ Path = "kaggle_notebooks/exp001_nearest_neighbor"; Ref = "dmitriigluzdov/biohub-exp001-nearest-neighbor-sanity"; Message = "EXP-001 nearest-neighbor sanity" },
    @{ Path = "kaggle_notebooks/exp002_rule_based"; Ref = "dmitriigluzdov/biohub-exp002-rule-based-classical"; Message = "EXP-002 classical blob Hungarian gap baseline" },
    @{ Path = "kaggle_notebooks/exp003_clean_single_seed"; Ref = "dmitriigluzdov/biohub-exp003-clean-single-seed"; Message = "EXP-003 clean single-seed UNet transformer ILP" },
    @{ Path = "kaggle_notebooks/exp004_dual_seed_blend"; Ref = "dmitriigluzdov/biohub-exp004-dual-seed-logit-blend"; Message = "EXP-004 dual-seed logit blend DeepCenter" }
)

foreach ($kernel in $kernels) {
    kaggle kernels push -p $kernel.Path
}

Write-Host "Kernels launched. Submit each completed version with:"
foreach ($kernel in $kernels) {
    Write-Host "kaggle competitions submit $competition -k $($kernel.Ref) -f submission.csv -m `"$($kernel.Message)`""
}
