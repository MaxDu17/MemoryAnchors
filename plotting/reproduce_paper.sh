#!/usr/bin/env bash
# Regenerate the paper's LIBERO figures and tables from the shipped per-run eval results in results/
# (CPU only, a few minutes). Usage: bash plotting/reproduce_paper.sh [output_dir (default: figures)]
#
# Paper item                      -> output
#   Fig. 6 left  / Table 3        -> <out>/fig6_left_remove_anchors/, <out>/table3_remove_anchors.md
#   Fig. 6 right / Table 4        -> <out>/fig6_right_add_anchors/,   <out>/table4_add_anchors.md
#   Table 2 (NBT, LIBERO-Goal)    -> <out>/table2_baselines.md
#   Fig. 2A / Fig. 14             -> <out>/fig2a_fixed_vs_rand/
#   Fig. 2B                       -> <out>/fig2b_buffer_resampling.png
#   Fig. 3 (one run)              -> <out>/fig3_cream_cheese_order0.png
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
OUT="${1:-figures}"
mkdir -p "$OUT"
declare -A SPLIT=([Libero10]=libero_10 [LiberoGoal]=libero_goal [LiberoObject]=libero_object [LiberoSpatial]=libero_spatial)
SUITES=(Libero10 LiberoGoal LiberoObject LiberoSpatial)

echo "== 1/6 per-run metrics (results/*/<Suite>/<run>/generated_visuals/)"
for exp in results/*/; do
  for suite in "${SUITES[@]}"; do
    [ -d "$exp$suite" ] || continue
    python plotting/compute_metrics.py "$exp$suite" --batch-parent --libero_split "${SPLIT[$suite]}" --no-plots > /dev/null
  done
done

echo "== 2/6 high-conflict task pairs (selected on RandER 1% runs, task orders 0-2, as in the paper)"
for suite in "${SUITES[@]}"; do
  python plotting/high_conflict/find_conflict_pairs.py "results/RandER_1pct/$suite" --batch-parent \
    --libero_split "${SPLIT[$suite]}" --orders 0,1,2 > /dev/null
done
for exp in RandER_1pct AnchorER-10_1pct AnchorER-20_1pct; do
  python plotting/high_conflict/extract_pair_deltas.py results/RandER_1pct "results/$exp" --orders 0,1,2 > /dev/null
done

echo "== 3/6 anchor removal (Fig. 6 left, Table 3)"
REMOVE=(results/RandER_1000 results/RandER-1pct_1000 results/RandER-5pct_1000 results/RandER-10pct_1000)
python plotting/compare_methods.py --pool-per-task --palette remove --fig-width 7.2 --prefix removeanchors \
  --experiment-dirs "${REMOVE[@]}" --labels RandER "RandER -1%" "RandER -5%" "RandER -10%" \
  --output-dir "$OUT/fig6_left_remove_anchors" > /dev/null
python plotting/metrics_table.py --experiment-dirs "${REMOVE[@]}" \
  --labels "100% (RandER)" "99% (RandER-1%)" "95% (RandER-5%)" "90% (RandER-10%)" > "$OUT/table3_remove_anchors.md"

echo "== 4/6 anchor addition (Fig. 6 right = bwt_bar_targettasks, Table 4)"
ADD=(results/RandER_1pct results/AnchorER-10_1pct results/AnchorER-20_1pct)
python plotting/compare_methods.py --pool-per-task --palette add --prefix addanchors \
  --experiment-dirs "${ADD[@]}" --labels RandER "AnchorER 10%" "AnchorER 20%" \
  --output-dir "$OUT/fig6_right_add_anchors" > /dev/null
python plotting/metrics_table.py --experiment-dirs "${ADD[@]}" --labels RandER "AnchorER (10%)" "AnchorER (20%)" \
  > "$OUT/table4_add_anchors.md"

echo "== 5/6 baselines (Table 2, LIBERO-Goal)"
python plotting/metrics_table.py --suites LiberoGoal \
  --experiment-dirs results/AnchorER-10_1pct results/MIR_1pct results/GSS_1pct results/AnchorER-NoDisagreement_1pct results/RandER_1pct \
  --labels Ours MIR GSS Ours-ActionDis RandER > "$OUT/table2_baselines.md"

echo "== 6/6 FixedER vs RandER (Fig. 2A, Fig. 14), buffer resampling (Fig. 2B), one task curve (Fig. 3)"
python plotting/fixed_vs_rand_violins.py --experiment-dirs results/FixedER_1000 results/RandER_1000 \
  --labels FixedER RandER --output-dir "$OUT/fig2a_fixed_vs_rand" > "$OUT/fig2a_nbt_spread.txt"
python plotting/buffer_resampling_curves.py results/RandER_resampling/LiberoGoal/*/generated_visuals/task_to_series.json \
  --task put_the_cream_cheese_in_the_bowl -o "$OUT/fig2b_buffer_resampling.png" > /dev/null
RUN=$(ls -d results/RandER_1pct/LiberoGoal/*order00* | head -1)
python plotting/task_curve.py "$RUN/generated_visuals/task_to_series.json" --task put_the_cream_cheese_in_the_bowl \
  --x-values 0,1,2,3,4,5,6,7,8,9 -o "$OUT/fig3_cream_cheese_order0.png" > /dev/null

echo "Done. Tables:"
cat "$OUT/table3_remove_anchors.md" "$OUT/table4_add_anchors.md" "$OUT/table2_baselines.md"
