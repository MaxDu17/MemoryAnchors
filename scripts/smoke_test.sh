#!/usr/bin/env bash
# End-to-end installation check (~10-20 min on one GPU; the first run also builds the dataset cache).
#
#   bash scripts/smoke_test.sh <gpu_id>
#
# 1. trains `anchor_removal` on the first 3 LIBERO-Goal tasks for a few steps each (so Memory Anchor
#    selection runs at the 2nd and 3rd task boundary),
# 2. evaluates the last checkpoint with 2 short rollouts per task,
# 3. computes the continual-learning metrics for the run.
# Output: outputs/smoke_test/LiberoGoal/<run>/ (safe to delete).
set -euo pipefail
GPU=${1:-0}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$GPU}
export HYDRA_FULL_ERROR=1 NUMBA_DISABLE_JIT=1 TOKENIZERS_PARALLELISM=false
OUT="$REPO/outputs/smoke_test/LiberoGoal"

echo "== [1/3] training (3 tasks x 1 epoch x 5 steps)"
CUDA_VISIBLE_DEVICES=$GPU python memory_anchors/train.py \
  task=libero_goal method=anchor_removal task.task_order_index=0 \
  exp_name=smoke_test output_dir_prefix="$OUT" logging.mode=disabled \
  task.epochs_per_task=1 training.num_epochs=3 \
  training.max_train_steps=5 training.max_val_steps=2 training.val_every=1 \
  learning_algorithm.selection_cfg.partition_device=cpu \
  task.env_runner.n_test=2 task.env_runner.n_envs=2 task.env_runner.n_test_vis=1 task.env_runner.max_steps=40

RUN=$(ls -td "$OUT"/*_smoke_test | head -1)
CKPT=$(ls "$RUN"/checkpoints/[0-9]*.ckpt | sort -t/ -k2 -V | tail -1)
echo "== [2/3] evaluating $CKPT"
CUDA_VISIBLE_DEVICES=$GPU python memory_anchors/evaluate.py --checkpoint_path "$CKPT"
test -s "$CKPT.json"

echo "== [3/3] metrics"
python plotting/compute_metrics.py "$RUN" --libero_split libero_goal
echo "Smoke test passed: $RUN"
