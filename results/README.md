# Shipped LIBERO results

Per-run evaluation results of the LIBERO experiments in the paper, so that every LIBERO figure and
table can be regenerated on a CPU without training (`bash plotting/reproduce_paper.sh`).

Each run folder mirrors a real training run (`outputs/<exp>/<Suite>/<run>/`):

```
<experiment>/<Suite>/<run>/
  .hydra/config.yaml               the run's config (original codebase; `umi_day.*` = memory_anchors.*)
  task_boundaries.json             [{task_name, end_step}] for each finished task, in training order
  checkpoints/<global_step>.ckpt.json
                                   eval of that checkpoint: means_test/<suite>/<task>_mean_score =
                                   success rate over LIBERO's 50 init states (videos, per-seed
                                   rewards and multi-BDDL results removed to keep this small)
```

Run `python plotting/compute_metrics.py <experiment>/<Suite> --batch-parent --libero_split <suite>` to
create `generated_visuals/` (metrics) inside each run. The task order of a run is the `orderXX` in
its name (LIBERO's benchmark permutation index); every run uses `training.seed=3`.

## Experiments

| Folder | Method (paper name) | Buffer | Paper |
|---|---|---|---|
| `RandER_1000` | RandER | 1000 | Fig. 2A, Fig. 6 left, Table 3 |
| `RandER-1pct_1000`, `RandER-5pct_1000`, `RandER-10pct_1000` | anchor removal: RandER −1% / −5% / −10% | 1000 | Fig. 6 left, Table 3 |
| `RandER_1pct` | RandER (also the reference runs for the high-conflict pairs) | 1% | Fig. 6 right, Table 2, Table 4 |
| `AnchorER-10_1pct`, `AnchorER-20_1pct` | AnchorER with 10% / 20% anchors | 1% | Fig. 6 right, Table 2 ("Ours" = 10%), Table 4 |
| `FixedER_1000` | FixedER (LIBERO's reference ER: 1000 per past task, see below) | 1000 / task | Fig. 2A, Fig. 14 |
| `MIR_1pct`, `GSS_1pct` | MIR, GSS | 1% | Table 2 |
| `AnchorER-NoDisagreement_1pct` | Ours-ActionDis (no action-disagreement step) | 1% | Table 2 |
| `RandER_resampling` | RandER, same task-8 checkpoint, 4 buffer samples (seeds 1-4) | 1000 | Fig. 2B |

## Number of runs (= task orders)

| Folder | Long | Goal | Object | Spatial |
|---|---|---|---|---|
| RandER_1000 | 3 | 3 | 3 | 3 |
| RandER-1pct_1000 | 3 | 3 | 3 | 3 |
| RandER-5pct_1000 | 3 | 6 ¹ | 3 | 3 |
| RandER-10pct_1000 | 3 | 3 | 3 | 3 |
| RandER_1pct | 3 | 7 ² | 3 | 6 ² |
| AnchorER-10_1pct | 3 | 7 ² | 3 | 4 ² |
| AnchorER-20_1pct | 3 | 3 | 3 | 3 |
| FixedER_1000 | 2 ³ | 3 | 3 | 3 |
| MIR_1pct / GSS_1pct / AnchorER-NoDisagreement_1pct | – | 7 ² | – | – |
| RandER_resampling | – | 4 | – | – |

¹ orders 0-2 were launched twice; both copies are kept, as in the paper's Table 3.
² orders 0-5 and 7 (GSS: 0-6). The paper's high-conflict analysis (Fig. 6 right) and its LIBERO-Goal
  RandER numbers in Tables 2 and 4 use orders 0-2 only (`--orders 0,1,2`); everything else uses all runs.
³ the order-2 run is excluded (see below).

## Excluded runs (not shipped)

| Folder | Suite | Order | Reason |
|---|---|---|---|
| AnchorER-10_1pct | Long | 3, 4, 5 | training did not finish |
| AnchorER-10_1pct | Spatial | 3, 5 | training did not finish (9/10 tasks) |
| FixedER_1000 | Long | 2 | no evaluation of the final checkpoint (the original plotting code silently substituted the previous checkpoint for the last stage) |

## Notes

- **Actions are absolute end-effector poses**: all policies were trained on, and output, absolute
  EE position + rotation + gripper (`abs_actions`, converted from LIBERO's delta actions).
- **LIBERO-Long epochs**: the removal / addition runs train 80 epochs per task on LIBERO-Long; the
  `RandER_1000` and `FixedER_1000` LIBERO-Long runs used 50.
- **FixedER memory**: `FixedER_1000` follows LIBERO's reference ER and keeps 1000 samples *per past
  task* (up to 9000), while `RandER_1000` keeps 1000 in total. FixedER therefore had more memory, which
  if anything understates the RandER advantage in Fig. 2A.
- Re-running `plotting/reproduce_paper.sh` matches Table 3 and Fig. 6 left to two decimals and Fig. 6
  right / Table 4 to within ±0.01, with the exceptions listed in plotting/README.md.
