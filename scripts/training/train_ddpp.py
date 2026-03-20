"""DDPP-LB fine-tuning for GenMol.

Usage (from project root):
    python scripts/train_ddpp.py reward=qed beta=0.25 num_steps=5000
"""

import logging
import os
import sys

sys.path.insert(0, os.path.realpath("."))
sys.path.insert(0, os.path.join(os.path.realpath("."), "src"))

import hydra
from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@hydra.main(version_base="1.3", config_path="../../configs/finetune", config_name="ddpp")
def main(cfg: DictConfig):
    from genmol.finetune.ddpp import DDPPLBTrainer
    DDPPLBTrainer.run_from_config(cfg)


if __name__ == "__main__":
    main()
