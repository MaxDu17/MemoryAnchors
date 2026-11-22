# Memory Anchors for Continual Robot Learning

Code for the CoRL 2026 paper **Memory Anchors for Continual Robot Learning**
(Maximilian Du, Zhanyi Sun, Chen Xu, Paarth Shah, Masha Itkina, Shuran Song).
Website: [robot-adaptation.github.io/MemoryAnchors](https://robot-adaptation.github.io/MemoryAnchors)

A replay buffer of past data is the standard way to keep a robot policy from forgetting old tasks
while it learns new ones. We show that a small subset of the replayed data does most of the work.
These are the **Memory Anchors**: old-task data whose observations overlap with the new task's in
the policy's latent space, while the two tasks require different actions. This repository contains:

- **Anchor extraction** (Sec. 4, Alg. 1): latent overlap → action disagreement → retrieval.
- **Anchor removal** (Sec. 5.1): exclude the top 1/5/10% anchors before sampling a random buffer.
- **AnchorER** (Sec. 5.2): reserve part of the buffer for anchors.
- **Baselines:** RandER, FixedER (LIBERO's reference ER), MIR, GSS, and the no-action-disagreement ablation.
- **Infrastructure:** a multi-GPU launcher, a checkpoint evaluator, metric and plotting code, and the per-run metrics behind the paper's LIBERO figures.

Everything runs on the four LIBERO suites (Goal, Long/LIBERO-10, Object, Spatial) with a U-Net
diffusion policy trained from scratch. The π0.5 VLA experiments (Sec. 5.3) and the real-robot
experiments (Sec. 5.4) are **not** part of this release.

---

## Contents
1. [Installation](#1-installation)
2. [Data](#2-data)
3. [Check the installation](#3-check-the-installation)
4. [Starter experiment: removing Memory Anchors](#4-starter-experiment-removing-memory-anchors)
5. [All experiments](#5-all-experiments)
6. [Reference: workflow, outputs, metrics](#6-reference)
7. [Notes and caveats](#7-notes-and-caveats)
8. [Repository layout](#8-repository-layout)
9. [Citation and acknowledgements](#9-citation-and-acknowledgements)

---

## 1. Installation

Requirements: Linux, an NVIDIA GPU with EGL support (headless MuJoCo rendering), conda.

```bash
git clone https://github.com/<org>/MemoryAnchors.git
cd MemoryAnchors
conda env create -f environment.yml      # ~10 min; installs torch (CUDA 12.8 wheels) + the vendored packages
conda activate memory_anchors
```

- **Older CUDA driver:** swap the two `torch` lines in `environment.yml` (see the comment at its top).
- **No submodules:** LIBERO, robomimic and the needed part of `diffusion_policy` are vendored under
  `third_party/`, each with its license and a list of our patches ([third_party/README.md](third_party/README.md)).
- **No `mujoco_py`:** it is not needed.
- **LIBERO config:** LIBERO's path config is written to `third_party/LIBERO/.libero/` on first import, not `~/.libero`.
- **Hugging Face models:** the sentence-T5 language encoder (`sentence-transformers/sentence-t5-base`) is
  downloaded on first use.
- **Weights & Biases:** logging is `offline` by default. Use `logging.mode=online` (or
  `--wandb online` in the launcher) to sync.

## 2. Data

```bash
python scripts/download_libero.py             # official LIBERO demos from Hugging Face -> data/libero/<suite>/  (~49 GB)
python scripts/convert_to_abs_actions.py      # -> data/libero_abs/<suite>/  (CPU, ~1 h per suite)
```

> **All policies in this repository (and all results in the paper) use absolute end-effector actions, not LIBERO's delta actions.**
> `convert_to_abs_actions.py` replays each demonstration in the simulator and stores the OSC
> controller's goal pose for every delta action as `data/demo_<i>/abs_actions`:
> position (3), axis-angle rotation (3) and gripper (1). The policy predicts position + 6D rotation +
> gripper (10 dims), and the environment is driven with absolute pose targets at evaluation.
> The conversion is deterministic except at the first step of each demonstration. There the
> controller's goal depends on state left over from the previous reset, so re-converted files match
> ours exactly from step 1 onward.

To keep data elsewhere, set `MEMORY_ANCHORS_DATA=/path/to/data` (the folder that contains `libero_abs/`).
`MEMORY_ANCHORS_CACHE` controls where the zarr dataset cache is written (default `cache/`, several GB per suite).

## 3. Check the installation

```bash
bash scripts/smoke_test.sh 0       # GPU 0; ~10-20 min (the first run also builds the dataset cache)
```

The smoke test does three things:
1. Trains `anchor_removal` on the first three LIBERO-Goal tasks for a few steps each, so Memory
   Anchors are selected at two task boundaries.
2. Evaluates a checkpoint with two short rollouts per task.
3. Computes the metrics.

It writes to `outputs/smoke_test/`.

## 4. Starter experiment: removing Memory Anchors

This reproduces the subtraction experiment (Sec. 5.1, Fig. 6 left, Table 3) on **LIBERO-Goal**. A
RandER buffer of 1000 transitions is compared against the same buffer sampled after the top 1%, 5% or
10% Memory Anchors are removed from the past data. This is the cleanest demonstration of the
anchors' effect (see [notes](#7-notes-and-caveats)).

**Cost (LIBERO-Goal):**
- 4 conditions × 3 task orders = 12 runs, each about 20 GPU-hours of training (≈9 GB of GPU memory).
- Plus 11 checkpoint evaluations per run, each about 40 min (≈7.5 GB, 10 parallel simulators).
- LIBERO-Long runs take about 3× longer.

### 4.1 Train

```bash
# runs are queued and started whenever one of the listed GPUs has 9 GB free (several runs can share a GPU)
python scripts/launch.py --name RandER_1000        --method rand_er        --buffer 1000 --suites goal --orders 0,1,2 --gpus 0,1,2,3
python scripts/launch.py --name RandER-1pct_1000   --method anchor_removal --buffer 1000 --suites goal --orders 0,1,2 --gpus 0,1,2,3 \
    --set learning_algorithm.selection_cfg.invert_remove_proportion=0.01
python scripts/launch.py --name RandER-5pct_1000   --method anchor_removal --buffer 1000 --suites goal --orders 0,1,2 --gpus 0,1,2,3 \
    --set learning_algorithm.selection_cfg.invert_remove_proportion=0.05
python scripts/launch.py --name RandER-10pct_1000  --method anchor_removal --buffer 1000 --suites goal --orders 0,1,2 --gpus 0,1,2,3 \
    --set learning_algorithm.selection_cfg.invert_remove_proportion=0.10
```

(`SUITES=goal GPUS=0,1,2,3 bash experiments/anchor_removal.sh` runs all four, then the evaluation.)
Each `launch.py` call blocks until its runs finish. To use more GPUs at once, run the four commands
in separate terminals. Each run gets its own Hydra directory:

```
outputs/RandER-10pct_1000/LiberoGoal/2026.11.22-10.00.00_RandER-10pct_1000_libero_goal_order00/
├── .hydra/config.yaml               # full resolved config (used again by the evaluator)
├── checkpoints/0.ckpt, 19749.ckpt…  # eval checkpoints: untrained model + end of every task
├── checkpoints/task1_<task>.ckpt…   # task-end checkpoints (for analysis / fine-tuning)
├── task_boundaries.json             # which task ended at which global step
├── selected_indices_<task>.npy      # ER buffer chosen at the start of <task> (indices into past data)
├── ood_indices_<task>.npy           # new-task samples with high action disagreement (step 2)
├── retrieval_dict_<task>.npy        # new-task sample -> retrieved old samples (step 3)
├── vis/                             # anchor / disagreement image grids and histograms per task
└── logs.json.txt, train.log
```

### 4.2 Evaluate

```bash
python scripts/evaluate_runs.py --root outputs --gpus 0,1,2,3 [--watch] [--delete-checkpoints-after-success]
```

The evaluator works as follows:
- It rolls out every `checkpoints/<global_step>.ckpt` on **all ten tasks** of the suite, using the
  50 LIBERO benchmark init states per task.
- It writes `<global_step>.ckpt.json` next to each checkpoint.
- Already-evaluated checkpoints are skipped.
- With `--watch` it keeps evaluating new checkpoints while training is still running.

Each checkpoint is ~0.7 GB, so `--delete-checkpoints-after-success` is useful once its json exists.

### 4.3 Metrics and plots

PLOTTING_STARTER_PLACEHOLDER

**Expected result** (paper, Table 3, LIBERO-Goal; mean ± SEM over 3 orders × 9 tasks). Removing
just 10% of the anchors raises forgetting (NBT) about 8×, while the ability to learn each new
task (FWT) is unchanged:

| Access to Memory Anchors | FWT ↑ | NBT ↓ | AUC ↑ |
|---|---|---|---|
| 100% (RandER) | 0.93 ± 0.01 | 0.05 ± 0.01 | 0.90 ± 0.01 |
| 99% (RandER −1%) | 0.93 ± 0.01 | 0.07 ± 0.01 | 0.87 ± 0.02 |
| 95% (RandER −5%) | 0.93 ± 0.01 | 0.20 ± 0.03 | 0.79 ± 0.02 |
| 90% (RandER −10%) | 0.92 ± 0.01 | 0.40 ± 0.05 | 0.64 ± 0.04 |

Expect differences of a few points between reruns: the simulator, data loading and anchor selection
are not bit-for-bit deterministic.

## 5. All experiments

Each paper result has a script in [experiments/](experiments/). Settings shared across scripts
(`GPUS`, `SUITES`, `ORDERS`, `MEM`) are read from the environment. Runs shared between experiments
(e.g. `RandER_1000`) are trained only once.

| Paper | Script | Runs (`--name`: method preset, buffer) |
|---|---|---|
| Fig. 2A, Fig. 14 | `fixed_vs_rand.sh` | `FixedER_1000`: `fixed_er`, 1000 (per past task), and `RandER_1000`: `rand_er`, 1000 |
| Fig. 2B | `buffer_resampling.sh` | `RandER_resampling_seed{1..4}`: fine-tune task 9 of LIBERO-Goal from a RandER checkpoint, 4 buffer samples |
| Fig. 6 left, Table 3 | `anchor_removal.sh` | `RandER_1000`, `RandER-{1,5,10}pct_1000`: `anchor_removal`, 1000 |
| Fig. 6 right, Table 4 | `anchor_addition.sh` | `RandER_1pct`, and `AnchorER-{10,20}_1pct`: `anchor_er`, 1% of past data |
| Table 2 | `baselines.sh` | `MIR_1pct`, `GSS_1pct`, `AnchorER-NoDisagreement_1pct` (+ the two above), LIBERO-Goal |

**Method presets** (`method=` in [memory_anchors/config/method/](memory_anchors/config/method/)):

| Preset | Paper name | What it does |
|---|---|---|
| `rand_er` | RandER | Uniform transition-level buffer, re-sampled from all past data at every task. |
| `fixed_er` | FixedER | LIBERO's reference ER: a fixed prefix of each past task (episode-level), never re-sampled. |
| `anchor_removal` | RandER −k% | Remove the top `invert_remove_proportion` anchors from the past data, then sample uniformly. |
| `anchor_er` | AnchorER | `1 − random_proportion` of the buffer are anchors (0.9 → 10%, 0.8 → 20%), the rest uniform. |
| `anchor_er_no_disagreement` | Ours-ActionDis | AnchorER without step 2: nearest neighbors of all new-task data. |
| `mir` | MIR | Random buffer; each step replays the candidates most interfered with by the current update. |
| `gss` | GSS | GSS-Greedy: keep past samples with diverse gradients. |

**Buffer size:** `--buffer 1000` means 1000 transitions. `--buffer 0.01` means 1% of all past-task
data, so the buffer grows as tasks accumulate. These map to `learning_algorithm.filter_size` and
`filter_size_mode`.

**Any other Hydra override** can be passed with `--set key=value`; the main knobs are in
[memory_anchors/config/learning_algorithm/anchor_er.yaml](memory_anchors/config/learning_algorithm/anchor_er.yaml).
To train a single run without the launcher:

```bash
MUJOCO_GL=egl python memory_anchors/train.py task=libero_goal method=anchor_er task.task_order_index=0 \
    learning_algorithm.filter_size=0.01 learning_algorithm.filter_size_mode=prop exp_name=my_run
```

### Reproducing the paper's plots without training

[results/](results/) contains the per-run evaluation summaries (task boundaries, per-checkpoint success
rates and the resolved configs) of the runs behind the paper's LIBERO figures and tables.

PLOTTING_RESULTS_PLACEHOLDER

## 6. Reference

**Workflow.** Each stage writes files that the next stage reads, so they can run on different machines:
1. `scripts/launch.py` (training) writes `checkpoints/<step>.ckpt` and `task_boundaries.json`.
2. `scripts/evaluate_runs.py` (simulator rollouts) writes `checkpoints/<step>.ckpt.json`.
3. `plotting/compute_metrics.py` (per-run metrics) writes `generated_visuals/continual_metrics.json`.
4. `plotting/compare_methods.py` (method × suite tables and plots) produces the final comparisons.

**Task orders.** `task.task_order_index` selects one of LIBERO's 21 benchmark permutations of each
suite. The paper uses orders 0, 1 and 2, and all paper runs use `training.seed=3`; run-to-run variation
therefore comes from the task order.

**Evaluation output** (`<step>.ckpt.json`, JSON-lines; read the last line). Keys are prefixed with
`test/<suite>/` or `means_test/<suite>/`. Task names are LIBERO's original BDDL names, e.g.
`open_the_top_layer_of_the_drawer_and_put_the_bowl_inside`; the mapping to the dataset file names
is in `memory_anchors/task_name_mappings/`.
- `means_test/<suite>/<task>_mean_score`: success rate on that task (50 rollouts).
- `test/<suite>/<task>_sim_max_reward_<100000+i>`: success (0/1) of rollout `i`.
- `test/<suite>/<task>_multi_bddl_success_<100000+i>`: which *other* tasks of the suite the final state satisfies.
- `test_mean_score`: average over all tasks of the suite, *including tasks not trained yet*. It is not a continual-learning metric.

**Metrics** (App. B.4). Let c<sub>i,k</sub> be the success on task *i* after training up to task *k*,
over N tasks:
- **NBT** (negative backward transfer, *forgetting*, lower is better): NBT<sub>i</sub> = mean over k>i of (c<sub>i,i</sub> − c<sub>i,k</sub>), averaged over tasks i < N.
- **FWT** (here: success right after learning the task): FWT<sub>i</sub> = c<sub>i,i</sub>.
- **AUC**: AUC<sub>i</sub> = mean over k ≥ i of c<sub>i,k</sub>.
- **NBT High Conflict** (Fig. 6 right): Δ<sub>i,k</sub> = c<sub>i,k−1</sub> − c<sub>i,k</sub> on the six task pairs with the largest forgetting under the reference RandER runs.

## 7. Notes and caveats

- **Absolute actions.** All training data, policies and results use absolute end-effector pose
  actions converted from LIBERO's deltas (Sec. 2 above).
- **The strongest evidence is the subtraction experiment.** RandER samples uniformly, so its buffer
  already contains Memory Anchors by chance. Removing them (Sec. 5.1) increases forgetting
  consistently on every suite. Adding more (AnchorER, Sec. 5.2) helps only in certain cases:
  - It reduces forgetting on the highest-conflict task pairs.
  - It clearly improves the average metrics on the homogeneous LIBERO-Goal suite.
  - Elsewhere the average gains are small (LIBERO-Spatial) or within noise (Long, Object; Table 4),
    because on heterogeneous suites the anchors that RandER samples by chance are usually enough.
- **NBT High Conflict follows the paper.** The six conflicting pairs are identified on the reference
  RandER runs and then scored for every method, including those same RandER runs. Pairs selected
  this way are, by construction, RandER's worst cases. The appendix (Table 4) therefore also
  reports the standard average metrics.
- **FixedER memory.** `fixed_er` defaults to `sequential_budget=per_task`, LIBERO's reference ER,
  which keeps `filter_size` transitions *per past task*. The paper's FixedER-1000 runs (Fig. 2) used
  this setting, so FixedER's buffer grew to up to 9000 transitions while RandER's stayed at 1000. This
  favors FixedER, so the comparison is conservative. Set
  `learning_algorithm.selection_cfg.sequential_budget=total` for an equal-memory comparison. In
  `prop` mode the two settings coincide.
- **LIBERO-Long epochs.** LIBERO-Long demonstrations are longer; the config trains 80 epochs per task
  (the other suites use 50). The paper's RandER-1000 and FixedER-1000 LIBERO-Long runs (Fig. 2 and
  the RandER bar of Fig. 6 left) used 50 epochs per task.
- **Anchor selection runs in train mode.** As in the paper, latents and action disagreement are
  computed with the model in training mode, so the train-time random crop and color jitter are
  active. Selection is therefore slightly stochastic.
  `learning_algorithm.selection_cfg.selection_eval_mode=true` uses eval mode instead.
- **Action disagreement** is the L2 position error plus 0.1× the rotation geodesic error between the
  denoised and the ground-truth action chunk, averaged over the chunk (gripper excluded). Noise is
  added at DDIM training timestep `noise_level=20` (of 50) and denoised with the 16-step inference
  schedule.
- **Normalization** statistics are computed once, on the data of all tasks of the suite (the same
  for every method).
- **Evaluation horizon** is 500 steps for every suite (not LIBERO's per-suite default).

## 8. Repository layout

```
memory_anchors/            the package
  algos/anchor_er.py       AnchorER / anchor removal / RandER / FixedER / NN ablation
  algos/mir.py, gss.py     MIR and GSS baselines
  workspace/               training loop (train_workspace.py) and checkpoint evaluation (eval_workspace.py)
  policy/, model/          diffusion policy and observation encoder
  dataset/, env_runner/    LIBERO data loading and simulator rollouts
  config/                  Hydra configs: train.yaml, task/, model/, learning_algorithm/, method/
  train.py, evaluate.py    entry points
scripts/                   launch.py, evaluate_runs.py, data download/conversion, smoke test
experiments/               one script per paper experiment
plotting/                  metrics and figures
analysis/                  anchor_concentration.py (Table 2, needs GPU + task checkpoints)
results/                   shipped per-run metrics of the paper's LIBERO runs
third_party/               vendored LIBERO, robomimic, diffusion_policy (see third_party/README.md)
```

## 9. Citation and acknowledgements

```bibtex
@inproceedings{du2026memoryanchors,
  title     = {Memory Anchors for Continual Robot Learning},
  author    = {Du, Maximilian and Sun, Zhanyi and Xu, Chen and Shah, Paarth and Itkina, Masha and Song, Shuran},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

This code builds on [Diffusion Policy](https://github.com/real-stanford/diffusion_policy),
[UMI](https://github.com/real-stanford/universal_manipulation_interface),
[robomimic](https://github.com/ARISE-Initiative/robomimic) and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO); see [third_party/README.md](third_party/README.md)
for the vendored versions, their licenses and our modifications.
