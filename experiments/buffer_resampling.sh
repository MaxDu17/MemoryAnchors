#!/usr/bin/env bash
# Sec. 3.1 / Fig. 2B -- same checkpoint, different random ER buffers.
# Starts from the RandER-1000 LIBERO-Goal (order 0) checkpoint taken after task 8 (Turn on the
# Stove), trains task 9 (Bowl to Plate) with 4 seeds (= 4 different buffer samples) and tracks the
# forgetting of Cream Cheese to Bowl. Needs a finished outputs/RandER_1000 LIBERO-Goal order-0 run.
source "$(dirname "$0")/_common.sh"

CKPT=${CKPT:-$(ls -d outputs/RandER_1000/LiberoGoal/*_libero_goal_order00 | head -1)/checkpoints/task8_turn_on_the_stove_demo.ckpt}
test -f "$CKPT" || { echo "Checkpoint not found: $CKPT (run experiments/anchor_removal.sh first or set CKPT=...)"; exit 1; }

for seed in 1 2 3 4; do
  launch --name "RandER_resampling_seed$seed" --method rand_er --buffer 1000 --suites goal_resample --orders 0 \
    --set "training.seed=$seed" --set "training.pretrained_checkpoint_path=$CKPT"
done
for seed in 1 2 3 4; do evaluate --root "outputs/RandER_resampling_seed$seed"; done

# Fig. 2B: success of Cream Cheese to Bowl while training Bowl to Plate, one curve per buffer sample
metrics RandER_resampling_seed1 RandER_resampling_seed2 RandER_resampling_seed3 RandER_resampling_seed4
python plotting/buffer_resampling_curves.py outputs/RandER_resampling_seed*/LiberoGoal/*/generated_visuals/task_to_series.json \
  --task put_the_cream_cheese_in_the_bowl -o figures/buffer_resampling.png
