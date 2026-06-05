#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion.pipeline import run_pipeline  # noqa: E402


@hydra.main(version_base=None, config_path="../config/diffusion", config_name="train")
def main(cfg: DictConfig) -> None:
    if bool(cfg.pipeline.get("dry_run", False)):
        print("[config]")
        print(OmegaConf.to_yaml(cfg, resolve=True))
    summary = run_pipeline(cfg)
    if bool(cfg.pipeline.get("dry_run", False)):
        print("[done]")
        print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


if __name__ == "__main__":
    main()
