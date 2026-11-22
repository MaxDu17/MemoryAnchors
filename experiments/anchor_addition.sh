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

# Table 4 and Fig. 6 (right): high-conflict pairs are picked on the RandER runs (orders 0-2), as in the paper
metrics RandER_1pct AnchorER-10_1pct AnchorER-20_1pct
for suite in "${!SPLIT[@]}"; do
  [ -d "outputs/RandER_1pct/$suite" ] || continue
  python plotting/high_conflict/find_conflict_pairs.py "outputs/RandER_1pct/$suite" --batch-parent \
    --libero_split "${SPLIT[$suite]}" --orders 0,1,2
done
for exp in RandER_1pct AnchorER-10_1pct AnchorER-20_1pct; do
  python plotting/high_conflict/extract_pair_deltas.py outputs/RandER_1pct "outputs/$exp" --orders 0,1,2
done
EXPS=(outputs/RandER_1pct outputs/AnchorER-10_1pct outputs/AnchorER-20_1pct)
python plotting/metrics_table.py --experiment-dirs "${EXPS[@]}" --labels RandER "AnchorER (10%)" "AnchorER (20%)"
python plotting/compare_methods.py --pool-per-task --palette add --prefix addanchors --experiment-dirs "${EXPS[@]}" \
  --labels RandER "AnchorER 10%" "AnchorER 20%" --output-dir figures/anchor_addition
