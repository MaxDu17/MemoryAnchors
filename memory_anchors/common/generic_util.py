import os

from omegaconf import OmegaConf

from memory_anchors import paths


def symlink_absolute(src, dest, **kwargs):
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    os.symlink(src, dest, **kwargs)


def symlink_relative(src, dest, **kwargs):
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    relative_src_path = os.path.relpath(src, start=os.path.dirname(dest))
    os.symlink(relative_src_path, dest, **kwargs)


def register_omegaconf_resolvers():
    """Resolvers used by the Hydra configs (safe to call more than once)."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("repo_root", paths.repo_root, replace=True)
    OmegaConf.register_new_resolver("data_root", paths.data_root, replace=True)
    OmegaConf.register_new_resolver("cache_root", paths.cache_root, replace=True)
    OmegaConf.register_new_resolver(
        "libero_bddl_dir", lambda suite: os.path.join(paths.libero_bddl_root(), suite), replace=True
    )
