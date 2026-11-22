#!/usr/bin/env bash
# Sec. 3.1 / Fig. 2A -- FixedER (LIBERO's reference ER) vs. RandER.
# FixedER uses sequential_budget=per_task (1000 examples per past task), as in the paper; pass
#   --set learning_algorithm.selection_cfg.sequential_budget=total
# for an equal-memory comparison. RandER_1000 is shared with anchor_removal.sh (completed runs are skipped).
source "$(dirname "$0")/_common.sh"

launch --name FixedER_1000 --method fixed_er --buffer 1000 --suites "$SUITES"
launch --name RandER_1000 --method rand_er --buffer 1000 --suites "$SUITES"

for name in FixedER_1000 RandER_1000; do
  evaluate --root "outputs/$name"
done
