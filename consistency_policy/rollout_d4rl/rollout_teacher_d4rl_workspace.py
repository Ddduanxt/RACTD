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

from consistency_policy.teacher_d4rl.edm_policy_d4rl import KarrasUnetHybridD4RLPolicy
from consistency_policy.base_workspace import BaseWorkspace
from consistency_policy.utils import load_normalizer

from contextlib import contextmanager
import time

OmegaConf.register_new_resolver("eval", eval, replace=True)

@contextmanager
def Timer(name):
    start_time = time.time()
    yield lambda: time.time() - start_time


class EDMRolloutD4RLWorkspace(BaseWorkspace):
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
        self.model: KarrasUnetHybridD4RLPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: KarrasUnetHybridD4RLPolicy = None
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
        
        # if not cfg.training.inference_mode:
        #     # configure dataset
        #     dataset: BaseLowdimDataset
        #     dataset = hydra.utils.instantiate(cfg.task.dataset)
        #     assert isinstance(dataset, BaseLowdimDataset)
        #     train_dataloader = DataLoader(dataset, **cfg.dataloader)
        #     normalizer = dataset.get_normalizer()

        #     # configure validation dataset
        #     # val_dataset = dataset.get_validation_dataset()
        #     # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        #     self.model.set_normalizer(normalizer)
        #     if cfg.training.use_ema:
        #         self.ema_model.set_normalizer(normalizer)

            # configure lr scheduler
            # lr_scheduler = get_scheduler(
            #     cfg.training.lr_scheduler,
            #     optimizer=self.optimizer,
            #     num_warmup_steps=cfg.training.lr_warmup_steps,
            #     num_training_steps=(
            #         len(train_dataloader) * cfg.training.num_epochs) \
            #             // cfg.training.gradient_accumulate_every,
            #     # pytorch assumes stepping LRScheduler every epoch
            #     # however huggingface diffusers steps it every batch
            #     last_epoch=self.global_step-1
            # )
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=100, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=5e-6, eps=1e-08)
        

        # # configure ema
        # ema: EMAModel = None
        # if cfg.training.use_ema:
        #     ema = hydra.utils.instantiate(
        #         cfg.ema,
        #         model=self.ema_model)

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
    workspace = EDMRolloutD4RLWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
