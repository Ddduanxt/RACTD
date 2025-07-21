if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

# from diffusion_policy.policy.diffusion_unet_lowdim_policy_maze2d import DiffusionUnetLowdimMazePolicy
from consistency_policy.teacher_maze.edm_policy_maze import KarrasUnetHybridMazePolicy

from consistency_policy.base_workspace import BaseWorkspace
from consistency_policy.utils import load_normalizer

from contextlib import contextmanager
import time

OmegaConf.register_new_resolver("eval", eval, replace=True)

@contextmanager
def Timer(name):
    start_time = time.time()
    yield lambda: time.time() - start_time


class EDMRolloutMazeWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        # cfg.policy.inference_mode = cfg.training.inference_mode
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: KarrasUnetHybridMazePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: KarrasUnetHybridMazePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training

        print(f"Loading from checkpoint {cfg.training.load_path}")
        self.load_checkpoint(path=cfg.training.load_path, exclude_keys=['optimizer'])
        workspace_state_dict = torch.load(cfg.training.load_path)
        normalizer = load_normalizer(workspace_state_dict)
        self.model.set_normalizer(normalizer)



        # configure env here
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        # timesteps = torch.arange(0, self.model.noise_scheduler.bins, device=device)
        # b, next_b = self.model.noise_scheduler.timesteps_to_times(timesteps[:-1]), self.model.noise_scheduler.timesteps_to_times(timesteps[1:])


        with JsonLogger(log_path) as json_logger:
            # for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            

            # ========= eval for this epoch ==========
            policy = self.model
            policy.eval()

            # run rollout --- here we interface with the simulator itself via env_runner
            runner_log = env_runner.run(policy)
            # log all
            step_log.update(runner_log)

            # end of epoch
            # log of last step is combined with validation and rollout
            print('step_log: ')
            print(step_log)
            wandb_run.log(step_log, step=self.global_step)
            json_logger.log(step_log)
            # self.global_step += 1
            # self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = EDMRolloutMazeWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
