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


class EDMMazeWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        cfg.policy.inference_mode = cfg.training.inference_mode
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
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            if cfg.training.resume_path != "None":
                print(f"Resuming from checkpoint {cfg.training.resume_path}")
                self.load_checkpoint(path=cfg.training.resume_path, exclude_keys=['optimizer'])
                workspace_state_dict = torch.load(cfg.training.resume_path)
                normalizer = load_normalizer(workspace_state_dict)
                self.model.set_normalizer(normalizer)
                
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file() and cfg.training.resume_path == "None":
                print(f"Resuming from checkpoint {lastest_ckpt_path}", exclude_keys=['optimizer'])
                self.load_checkpoint(path=lastest_ckpt_path)
                workspace_state_dict = torch.load(lastest_ckpt_path)
                normalizer = load_normalizer(workspace_state_dict)
                self.model.set_normalizer(normalizer)

        if not cfg.training.inference_mode:
            # configure dataset
            dataset: BaseLowdimDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseLowdimDataset)
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            normalizer = dataset.get_normalizer()

            # configure validation dataset
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

            # configure lr scheduler
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(
                    len(train_dataloader) * cfg.training.num_epochs) \
                        // cfg.training.gradient_accumulate_every,
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=self.global_step-1
            )
        

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        if cfg.training.online_rollouts:
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


        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        timesteps = torch.arange(0, self.model.noise_scheduler.bins, device=device)
        b, next_b = self.model.noise_scheduler.timesteps_to_times(timesteps[:-1]), self.model.noise_scheduler.timesteps_to_times(timesteps[1:])


        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                if not cfg.training.inference_mode:
                    # ========= train for this epoch ==========
                    train_losses = list()
                    with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            # compute loss
                            raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()
                            
                            # update ema
                            if cfg.training.use_ema:
                                ema.step(self.model)

                            # logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }

                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                                self.global_step += 1

                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                    # at the end of each epoch
                    # replace train_loss with epoch average
                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout --- here we interface with the simulator itself via env_runner
                if (self.epoch % cfg.training.rollout_every) == 0 and cfg.training.online_rollouts:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if not cfg.training.inference_mode:
                    # t_time = 0
                    # count = 0
                    # if (self.epoch % cfg.training.val_every) == 0:
                    #     with torch.no_grad():
                    #         val_losses = list()
                    #         val_mse_error = list()
                    #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                    #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    #             for batch_idx, batch in enumerate(tepoch):
                    #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    #                 loss = self.model.compute_loss(batch)
                                    
                    #                 if (self.epoch % cfg.training.val_sample_every) == 0:
                    #                     obs_dict = {'obs': batch['obs']}
                    #                     gt_action = batch['action']

                                        
                    #                     start_time = time.time()
                    #                     result = policy.predict_action(obs_dict)
                    #                     t = time.time() - start_time

                    #                     t_time += t
                    #                     count += 1
                                        
                    #                     pred_action = result['action_pred']
                    #                     mse = torch.nn.functional.mse_loss(pred_action, gt_action)


                    #                     val_losses.append(loss)
                    #                     val_mse_error.append(mse.item())
                    #                     del obs_dict
                    #                     del gt_action
                    #                     del result
                    #                     del pred_action
                    #                     del mse
                    #                 if (cfg.training.max_val_steps is not None) \
                    #                     and batch_idx >= (cfg.training.max_val_steps-1):
                    #                     break

                    #         if len(val_losses) > 0:
                    #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                    #             step_log['val_loss'] = val_loss
                                
                    #         if len(val_mse_error) > 0:
                    #             val_mse_error = torch.mean(torch.tensor(val_mse_error)).item()
                    #             step_log['val_mse_error'] = val_mse_error

                    #             val_avg_inference_time = t_time / count
                    #             step_log['val_avg_inference_time'] = val_avg_inference_time

                    # run diffusion sampling on a training batch
                    t_time = 0
                    count = 0
                    if (self.epoch % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['observations']
                            # gt_action = batch['actions']
                            
                            
                            start_time = time.time()
                            result = policy.predict_action(batch)
                            t = time.time() - start_time

                            t_time += t
                            count += 1
                            

                            pred_observations = result['observations_pred']

                            mse = torch.nn.functional.mse_loss(pred_observations, obs_dict)
                            step_log['train_mse_error'] = mse.item()
                            
                            pred_observations = pred_observations.cpu().numpy()
                            obs_dict = obs_dict.cpu().numpy()

                            savepath = os.path.join(self.output_dir, f'{self.epoch}_sample-reference.png')
                            env_runner.renderer.composite(savepath, obs_dict[:8])

                            savepath = os.path.join(self.output_dir, f'{self.epoch}_sample-generate.png')
                            env_runner.renderer.composite(savepath, pred_observations[:8])

                            
                            # step_log['train_avg_inference_time'] = t_time / count
                            del batch
                            del obs_dict
                            # del gt_action
                            del result
                            del pred_observations
                            del mse
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                print('step_log: ')
                print(step_log)
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = EDMMazeWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
