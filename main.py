import os
import random

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from train import train_entrypoint


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if config.train.fsdp:
        rank = os.environ["LOCAL_RANK"]
        world_size = os.environ["WORLD_SIZE"]
    else:
        rank = 0
        world_size = torch.cuda.device_count()

    train_entrypoint(rank, world_size, config)


if __name__ == "__main__":
    main()
