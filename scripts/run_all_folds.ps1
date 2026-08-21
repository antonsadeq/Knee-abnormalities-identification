# Train all five folds sequentially, then evaluate out-of-fold. PowerShell counterpart of
# run_all_folds.sh, which needs bash.
#
# The five folds must share one configuration. Mixing them makes the rank-ensemble incoherent and the
# out-of-fold estimate a blend of models trained against different objectives — so this script, not a
# hand-typed command line, is what defines "the" config.
#
#   .\scripts\run_all_folds.ps1                       # defaults below
#   .\scripts\run_all_folds.ps1 -Compile -Workers 4   # if preflight said compile works

param(
  [double]$MemoryFraction = 0.85,   # a dedicated 24 GB card; preflight prints the right number
  [int]$Workers = 4,                # Windows spawns workers, so start low and raise if the GPU idles
  [int]$Slices = 16,
  [int]$Epochs = 24,
  [switch]$Compile
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$env:PYTHONPATH = "$root\src"

Write-Host "=== verifying the cache before spending hours on it ==="
python "$root\scripts\verify_local.py"
if ($LASTEXITCODE -ne 0) {
  Write-Host "verify_local.py failed — a partial cache trains quietly and scores lower. Stopping." -ForegroundColor Red
  exit 1
}

$extra = @()
if ($Compile) { $extra += "--compile" }

foreach ($fold in 0..4) {
  $tag = "M4_f${fold}_convnext_tiny_${Slices}sl"
  if (Test-Path "$root\artifacts\runs\$tag.pt") {
    Write-Host "### fold $fold already done, skipping"
    continue
  }
  Write-Host "### fold $fold starting $(Get-Date -Format HH:mm:ss)"
  python "$root\scripts\train.py" `
    --fold $fold --epochs $Epochs --batch 2 --accum 4 --slices $Slices `
    --drop-path 0.1 --slot-dropout 0.1 --patience 5 `
    --ema 0.999 --tta `
    --memory-fraction $MemoryFraction --workers $Workers `
    --backbone convnext_tiny --log-every 100 `
    --tag $tag --out "$root\artifacts\runs" @extra `
    *> "$root\artifacts\runs\$tag.log"
  $best = Select-String -Path "$root\artifacts\runs\$tag.log" -Pattern "^best" | Select-Object -Last 1
  Write-Host "### fold $fold done $(Get-Date -Format HH:mm:ss): $best"
}

Write-Host "### all folds finished — evaluating out-of-fold over all 58 gold studies"
python "$root\scripts\evaluate_oof.py" --tag-pattern "M4_f{fold}_convnext_tiny_${Slices}sl" `
  --batch 2 --workers $Workers --memory-fraction 0.40 `
  *> "$root\artifacts\runs\M4_oof.log"
Get-Content "$root\artifacts\runs\M4_oof.log" -Tail 30
