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
import math
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import sklearn
from contextlib import contextmanager
import time

from consistency_policy.student_d4rl.ctm_policy_d4rl import CTMPUnetD4RLPolicy
from consistency_policy.base_workspace import BaseWorkspace
from consistency_policy.utils import load_normalizer

OmegaConf.register_new_resolver("eval", eval, replace=True)




class RolloutD4RLWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        cfg.policy.inference_mode = cfg.training.inference_mode
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: CTMPUnetD4RLPolicy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):

        cfg = copy.deepcopy(self.cfg)
            
        print(f"Loading from checkpoint {cfg.training.load_path}")
        self.load_checkpoint(path=cfg.training.load_path, exclude_keys=['optimizer'])
        workspace_state_dict = torch.load(cfg.training.load_path)
        normalizer = load_normalizer(workspace_state_dict)
        self.model.set_normalizer(normalizer) 

        # configure env
        self.output_dir = cfg.training.output_dir
        if cfg.training.online_rollouts:
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
        wandb.run.log_code(".")


        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)


        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        if cfg.training.inference_mode:
            self.model.drop_teacher()
        
        with JsonLogger(log_path) as json_logger:
            step_log = dict()

            # ========= eval ==========
            policy = self.model
            if cfg.training.use_ema:
                policy.use_ema = True
            policy.eval()

            # run rollout
            ## enable chaining
            # policy.enable_chaining()

            policy.chaining_steps = cfg.training.val_chaining_steps
            runner_log = env_runner.run(policy)
            policy.chaining_steps = 1

            # log all
            step_log.update(runner_log)
            wandb_run.log(step_log, step=self.global_step)
            json_logger.log(step_log)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = RolloutD4RLWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
