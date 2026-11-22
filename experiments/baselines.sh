#!/usr/bin/env bash
# Sec. 5.5 / Table 2 -- MIR, GSS and the no-action-disagreement ablation on LIBERO-Goal (1% buffer).
# RandER_1pct and AnchorER-10_1pct are shared with anchor_addition.sh.
source "$(dirname "$0")/_common.sh"

launch --name MIR_1pct --method mir --buffer 0.01 --suites goal
launch --name GSS_1pct --method gss --buffer 0.01 --suites goal
launch --name AnchorER-NoDisagreement_1pct --method anchor_er_no_disagreement --buffer 0.01 --suites goal
launch --name RandER_1pct --method rand_er --buffer 0.01 --suites goal
launch --name AnchorER-10_1pct --method anchor_er --buffer 0.01 --suites goal \
  --set learning_algorithm.selection_cfg.random_proportion=0.9

for name in MIR_1pct GSS_1pct AnchorER-NoDisagreement_1pct RandER_1pct AnchorER-10_1pct; do
  evaluate --root "outputs/$name"
done
