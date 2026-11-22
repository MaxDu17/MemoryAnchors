# Plotting and analysis

All scripts are CPU-only (except `analysis/anchor_concentration.py`) and read the run layout written
by training + `scripts/evaluate_runs.py`:
`<experiment>/<Suite>/<run>/{.hydra/config.yaml, task_boundaries.json, checkpoints/<step>.ckpt.json}`.
`results/` ships exactly this layout for the paper's runs.

**Regenerate every LIBERO figure/table of the paper:** `bash plotting/reproduce_paper.sh [out_dir]` (~2 min).

## Pipeline

1. **`compute_metrics.py`**: per run: success of every task at every checkpoint, the continual
   success matrix, FWT / NBT / AUC (paper App. B.4; `bwt` in the json = NBT, higher = more
   forgetting), task-task interactions Δ<sub>i,j</sub>, and `complete.txt` if the run finished its curriculum
   with an eval at every task boundary. Every later step only uses complete runs.
2. **`high_conflict/`** ("NBT High Conflict", Fig. 6 right / App. B.4), only for anchor addition:
   - `find_conflict_pairs.py`: the top-6 (source, trained) task pairs with the largest Δ on the
     reference RandER runs;
   - `extract_pair_deltas.py`: Δ of those pairs in every run of a method.
   As in the paper, the pairs are chosen on the same RandER runs that are then compared. This favors
   the other methods, because RandER is evaluated on its own worst pairs; the paper also reports
   standard NBT (Table 4).
3. **`compare_methods.py`**: method × suite bars (mean ± SEM) + json for fwt/bwt/auc, worst-case
   variants and `bwt_targettasks` (high conflict). **`metrics_table.py`**: Markdown tables.

## Paper figure / table → command

Paths are relative to the repo root; `reproduce_paper.sh` runs all of these.

| Paper | Script | Command (after step 1) |
|---|---|---|
| Fig. 6 left (removal) | `compare_methods.py` | `--pool-per-task --palette remove --experiment-dirs results/RandER_1000 results/RandER-{1,5,10}pct_1000` → `*bwt_bar.png` |
| Table 3 | `metrics_table.py` | `--experiment-dirs results/RandER_1000 results/RandER-1pct_1000 results/RandER-5pct_1000 results/RandER-10pct_1000` |
| Fig. 6 right (addition) | `high_conflict/*` + `compare_methods.py` | pairs with `--orders 0,1,2`, then `--pool-per-task --palette add --experiment-dirs results/RandER_1pct results/AnchorER-10_1pct results/AnchorER-20_1pct` → `*bwt_bar_targettasks.png` |
| Table 4 | `metrics_table.py` | `--experiment-dirs results/RandER_1pct results/AnchorER-10_1pct results/AnchorER-20_1pct` |
| Table 2 (NBT) | `metrics_table.py` | `--suites LiberoGoal --experiment-dirs results/AnchorER-10_1pct results/MIR_1pct results/GSS_1pct results/AnchorER-NoDisagreement_1pct results/RandER_1pct` |
| Table 2 (Anchor Conc.) | `analysis/anchor_concentration.py` (GPU; needs full training runs with checkpoints) | `<suite_dir> --anchor_select_proportion 0.1 --selected-sample-proportion 0.4` (MIR: add `--selected-sample-mode mir_topk`) |
| Fig. 2A / Fig. 14 | `fixed_vs_rand_violins.py` | `--experiment-dirs results/FixedER_1000 results/RandER_1000 --labels FixedER RandER` |
| Fig. 2B | `buffer_resampling_curves.py` | `results/RandER_resampling/LiberoGoal/*/generated_visuals/task_to_series.json` |
| Fig. 3 | `task_curve.py` | `<run>/generated_visuals/task_to_series.json --task put_the_cream_cheese_in_the_bowl` |

All comparison scripts accept `--orders 0,1,2` to restrict to specific task orders, and `--labels`.

## Reproduced vs. paper

Run `reproduce_paper.sh` to regenerate these numbers.

| Item | Reproduced | Paper | Note |
|---|---|---|---|
| Table 3 / Fig. 6 left | identical NBT except LIBERO-Goal −5% (0.19 vs 0.20); FWT/AUC within 0.01 | | the paper pooled two seeds for that cell; one is shipped (see results/README.md) |
| Fig. 6 right averages RandER → AnchorER 10% / 20% | 0.38 → 0.13 / 0.16 | 0.38 → 0.14 / 0.16 | identical to the paper's figure data |
| Table 4 (all cells except below) | within 0.01 | | |
| Table 4 / Table 2, LIBERO-Goal RandER NBT | 0.16 (7 orders) | 0.18 | the paper used orders 0-2 for this cell (`--orders 0,1,2` gives 0.18 / 0.79 AUC) |
| Table 2 Ours-ActionDis | 0.12 | 0.13 | |
| Fig. 2A pooled-NBT variance, FixedER vs RandER | 37% lower | 36% lower | |

## Not included

The π0.5 VLA results (Sec. 5.3, Table 1) are in
[MaxDu17/MemoryAnchors_VLA](https://github.com/MaxDu17/MemoryAnchors_VLA). Not included at all: the
real-robot results (OpenJar, SweaterFold), the buffer-size sweep (Fig. 12), and the qualitative
render / t-SNE tools.
