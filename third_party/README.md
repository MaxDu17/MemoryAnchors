# Vendored third-party code

These packages are copied into the repository (instead of git submodules) so that a plain
`git clone` is enough. Each keeps its original license file. They are installed in editable mode
by `environment.yml`.

| Package | Upstream | Version copied | What is included |
|---|---|---|---|
| `LIBERO/` | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT) | `8f1084e` + patches below | `libero/libero` only: environments, BDDL task files, init states, assets. `libero.lifelong` (LIBERO's own training code) is not included. |
| `robomimic/` | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) (MIT) | v0.2.0 + patches below | the `robomimic` package (docs/examples removed) |
| `diffusion_policy/` | [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) (MIT), which builds on [diffusion_policy](https://github.com/real-stanford/diffusion_policy) | our fork, commit `cd06ee5` | only the ~30 modules MemoryAnchors imports (U-Net, normalizer, EMA, workspace base, gym wrappers, ...) |

## Patches

**LIBERO**
- `libero/libero/envs/env_wrapper.py`: `ControlEnv.register_init_state(init_state)`. When set, `reset()` restores that simulator state instead of sampling a new one (used to replay benchmark init states).
- `libero/libero/envs/bddl_base_domain.py`: `check_success_for_goal_state()` / `check_success_multi_bddl()` evaluate the current state against the goals of *other* BDDL files of the same scene. The evaluator logs these as `multi_bddl_success_*` (e.g. "did the policy solve a different task instead?").
- `libero/libero/__init__.py`: the path config lives in `third_party/LIBERO/.libero/config.yaml` (override with `LIBERO_CONFIG_PATH`) and is written without the interactive `input()` prompt. Upstream defaults to a global `~/.libero/config.yaml`, which hangs non-interactive jobs on first use and can silently point to a different LIBERO checkout. There is also no warning about LIBERO's own (unused) datasets folder.
- `libero/__init__.py`, `setup.py`: minimal packaging for the trimmed copy.

**robomimic**
- `envs/env_robosuite.py`:
  - Supports robosuite >= 1.4 model-XML loading (`edit_model_xml`).
  - Removes the slow `egl_probe` device query.
  - Removes the hard dependency on `mujoco_py`, which was imported only to name an exception type; uses `mujoco.FatalError` instead.
- `setup.py`: minimal; dependencies are managed by `environment.yml` (`egl_probe` is no longer required).

**diffusion_policy**
- `policy/diffusion_unet_timm_policy.py` (from our UMI fork) adds the methods Memory Anchors needs:
  - `compute_ood_score`: action disagreement via noise-denoise.
  - `compute_obs_representations`: observation latents.
  - Per-sample losses (`reduce_loss=False`).
- `common/pose_util.py` is copied from `umi/common/pose_util.py` (the only UMI module needed); `model/common/rotation_transformer.py` imports it from here.
- `setup.py`, `__init__.py` files: minimal packaging. Unused modules were removed.
