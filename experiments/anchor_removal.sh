#!/usr/bin/env bash
# Sec. 5.1 / Fig. 6 (left) / Table 3 -- Removing Memory Anchors (the starter experiment).
# RandER with a 1000-example buffer vs. the same buffer sampled after excluding the top
# 1% / 5% / 10% Memory Anchors from the past data.
source "$(dirname "$0")/_common.sh"

launch --name RandER_1000 --method rand_er --buffer 1000 --suites "$SUITES"
for pct in 01 05 10; do
  launch --name "RandER-${pct#0}pct_1000" --method anchor_removal --buffer 1000 --suites "$SUITES" \
    --set "learning_algorithm.selection_cfg.invert_remove_proportion=0.$pct"
done

for name in RandER_1000 RandER-1pct_1000 RandER-5pct_1000 RandER-10pct_1000; do
  evaluate --root "outputs/$name"
done
