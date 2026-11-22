# Shared settings for the experiment scripts. Override from the environment, e.g.
#   GPUS=0,1,2,3 SUITES=goal bash experiments/anchor_removal.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPUS=${GPUS:-0}                       # GPUs to schedule runs/evals on
SUITES=${SUITES:-goal,10,object,spatial}
ORDERS=${ORDERS:-0,1,2}               # LIBERO task orders (paper: 0,1,2)
MEM=${MEM:-9000}                      # free GPU memory (MB) needed to start one training run
EVAL_MEM=${EVAL_MEM:-7500}            # ... and one evaluation
launch() { python scripts/launch.py --gpus "$GPUS" --mem-per-run-mb "$MEM" --orders "$ORDERS" "$@"; }
evaluate() { python scripts/evaluate_runs.py --gpus "$GPUS" --mem-per-eval-mb "$EVAL_MEM" "$@"; }

# Per-run metrics (generated_visuals/ inside every finished run) for the given experiment names.
declare -A SPLIT=([Libero10]=libero_10 [LiberoGoal]=libero_goal [LiberoObject]=libero_object [LiberoSpatial]=libero_spatial)
metrics() {
  for exp in "$@"; do
    for suite in "${!SPLIT[@]}"; do
      [ -d "outputs/$exp/$suite" ] || continue
      python plotting/compute_metrics.py "outputs/$exp/$suite" --batch-parent --libero_split "${SPLIT[$suite]}" --no-plots
    done
  done
}
