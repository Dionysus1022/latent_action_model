#!/usr/bin/env python3
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from diffusion.consistency_distillation import hydra_main


@hydra.main(version_base=None, config_path="./config/consistency", config_name="train")
def main(cfg: DictConfig) -> None:
    hydra_main(cfg)


if __name__ == "__main__":
    main()
