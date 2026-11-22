#!/usr/bin/env bash
# Sec. 5.2 / Fig. 6 (right) / Table 4 -- Adding Memory Anchors (AnchorER).
# 1% buffer (proportion of all past data): RandER vs. AnchorER with 10% / 20% of the buffer as anchors.
source "$(dirname "$0")/_common.sh"

launch --name RandER_1pct --method rand_er --buffer 0.01 --suites "$SUITES"
launch --name AnchorER-10_1pct --method anchor_er --buffer 0.01 --suites "$SUITES" \
  --set learning_algorithm.selection_cfg.random_proportion=0.9
launch --name AnchorER-20_1pct --method anchor_er --buffer 0.01 --suites "$SUITES" \
  --set learning_algorithm.selection_cfg.random_proportion=0.8

for name in RandER_1pct AnchorER-10_1pct AnchorER-20_1pct; do
  evaluate --root "outputs/$name"
done
