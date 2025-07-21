from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
# from consistency_policy.diffusion_unet_with_dropout import ValueUnet1D
from diffusion_policy.model.diffusion.conditional_unet1d import ValueUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetD4RLReward(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        
        obs_shape = shape_meta['observation']['shape']
        assert len(obs_shape) == 1
        obs_dim = obs_shape[0]

        # create reward model
        input_dim = action_dim + obs_dim

        model = ValueUnet1D(
            input_dim=input_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion reward params: %e" % sum(p.numel() for p in self.model.parameters()))


    def predict_reward(self, trajectory) -> Dict[str, torch.Tensor]:

        reward = self.model(trajectory)
        
        result = {
            'reward': reward
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer['observations'].normalize(batch['observations'])
        nactions = self.normalizer['actions'].normalize(batch['actions'])
        nrewards = self.normalizer['rewards'].normalize(batch['rewards'])

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        this_nobs = nobs.reshape(-1, *nobs.shape[2:])

        nactions = nactions.reshape(-1, *nactions.shape[2:])

        cond_data = torch.cat([nactions, this_nobs], dim=-1)
        trajectory = cond_data.detach()
        
        pred = self.model(trajectory)
        pred = pred.reshape(batch_size*horizon, 1)
        nrewards = nrewards.reshape(-1, 1)


        loss = F.mse_loss(pred, nrewards, reduction='none')
        loss = loss.mean()
        return loss
