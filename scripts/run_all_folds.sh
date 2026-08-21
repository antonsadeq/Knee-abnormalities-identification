#!/bin/bash
# Train all five folds with the final configuration, sequentially.
#
# Waits for any in-flight run to finish first. The folds must share one configuration or the
# ensemble is incoherent and the out-of-fold estimate mixes models that saw different objectives —
# so this is the script that defines "the" config, not five separate command lines.
set -u
R=/home/anton/projects/personal/DDR-developments/rsna-knee-abnormality-detection
PY=/home/anton/miniconda3/envs/rsna-knee/bin/python
cd "$R"
export PYTHONPATH="$R/src"

# Do not compete with a run already holding the GPU.
while pgrep -f "train[.]py --fold" > /dev/null; do sleep 120; done

for FOLD in 0 1 2 3 4; do
  TAG="M4_f${FOLD}_convnext_tiny_16sl"
  if [ -f "$R/artifacts/runs/${TAG}.pt" ]; then
    echo "### fold $FOLD already done, skipping"; continue
  fi
  echo "### fold $FOLD starting $(date +%H:%M:%S)"
  $PY "$R/scripts/train.py" \
    --fold "$FOLD" --epochs 24 --batch 2 --accum 4 --slices 16 \
    --drop-path 0.1 --slot-dropout 0.1 --patience 5 \
    --ema 0.999 --tta \
    --memory-fraction 0.55 --workers 12 --compile --backbone convnext_tiny \
    --log-every 100 --tag "$TAG" --out "$R/artifacts/runs" \
    > "$R/artifacts/runs/${TAG}.log" 2>&1
  echo "### fold $FOLD done $(date +%H:%M:%S): $(grep '^best' "$R/artifacts/runs/${TAG}.log" || echo FAILED)"
done

echo "### all folds finished $(date +%H:%M:%S)"
$PY "$R/scripts/evaluate_oof.py" --tag-pattern "M4_f{fold}_convnext_tiny_16sl" \
   --batch 2 --workers 4 --memory-fraction 0.20 \
   > "$R/artifacts/runs/M4_oof.log" 2>&1
echo "### OOF evaluation written to artifacts/runs/M4_oof.log"
