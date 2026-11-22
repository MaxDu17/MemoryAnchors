import os
import yaml

# [MemoryAnchors patch] Upstream LIBERO stores its path config in ~/.libero and
# prompts interactively (input()) on first import. That hangs non-interactive
# jobs, and a global ~/.libero/config.yaml can silently point to a *different*
# LIBERO checkout. We default to a config next to this vendored copy instead
# (third_party/LIBERO/.libero/config.yaml). Set LIBERO_CONFIG_PATH to override.
_VENDORED_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".libero"
)
libero_config_path = os.environ.get("LIBERO_CONFIG_PATH", _VENDORED_CONFIG_DIR)
config_file = os.path.join(libero_config_path, "config.yaml")


def get_default_path_dict(custom_location=None):
    if custom_location is None:
        benchmark_root_path = os.path.dirname(os.path.abspath(__file__))
    else:
        benchmark_root_path = custom_location

    # This is a default path for localizing all the default bddl files
    bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")

    # This is a default path for localizing all the default bddl files
    init_states_default_path = os.path.join(benchmark_root_path, "./init_files")

    # This is a default path for localizing all the default datasets
    dataset_default_path = os.path.join(benchmark_root_path, "../datasets")

    # This is a default path for localizing all the default assets
    assets_default_path = os.path.join(benchmark_root_path, "./assets")

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": bddl_files_default_path,
        "init_states": init_states_default_path,
        "datasets": dataset_default_path,
        "assets": assets_default_path,
    }


def get_libero_path(query_key):
    with open(config_file, "r") as f:
        config = dict(yaml.load(f.read(), Loader=yaml.FullLoader))

    # Give warnings in case the user needs to access the paths
    for key in config:
        if key == "datasets":  # [MemoryAnchors patch] LIBERO's own dataset folder is unused here
            continue
        if not os.path.exists(config[key]):
            print(f"[Warning]: {key} path {config[key]} does not exist!")

    assert (
        query_key in config
    ), f"Key {query_key} not found in config file {config_file}. You need to modify it. Available keys are: {config.keys()}"
    return config[query_key]


def set_libero_default_path(custom_location=os.path.dirname(os.path.abspath(__file__))):
    print(
        f"[Warning] You are changing the default path for Libero config. This will affect all the paths in the config file."
    )
    new_config = get_default_path_dict(custom_location)
    with open(config_file, "w") as f:
        yaml.dump(new_config, f)


def _write_default_config():
    """Non-interactively write the default path config (no input() prompt)."""
    os.makedirs(libero_config_path, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(get_default_path_dict(), f)


if not os.path.exists(config_file):
    _write_default_config()
elif libero_config_path == _VENDORED_CONFIG_DIR:
    # Repo moved or was re-cloned: refresh the vendored config so it never points
    # at stale absolute paths.
    with open(config_file, "r") as f:
        _cfg = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
    if _cfg.get("benchmark_root") != get_default_path_dict()["benchmark_root"]:
        _write_default_config()
