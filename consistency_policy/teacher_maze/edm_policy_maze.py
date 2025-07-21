from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from consistency_policy.diffusion import Karras_Scheduler, Huber_Loss
from consistency_policy.diffusion_unet_with_dropout import ConditionalUnet1D



class KarrasUnetHybridMazePolicy(BaseLowdimPolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: Karras_Scheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=32,
            down_dims=(64,128,256),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            delta=.0,
            inference_mode=False,
):
        super().__init__()

        # parse shape_meta
        # action_shape = shape_meta['action']['shape']
        # assert len(action_shape) == 1
        # action_dim = action_shape[0]

        obs_shape = shape_meta['observation']['shape']
        assert len(obs_shape) == 1
        obs_dim = obs_shape[0]


        # create diffusion model
        input_dim = obs_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = obs_dim
            global_cond_dim = obs_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        model.prepare_drop_generators()

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=obs_dim,
            obs_dim=0 if obs_as_global_cond else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = obs_dim
        self.n_action_steps = horizon
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond

        self.delta = delta

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = scheduler.sample_inital_position(condition_data, generator=generator)
    
        timesteps = torch.arange(0, self.noise_scheduler.bins, device=condition_data.device)
        for b, next_b in zip(timesteps[:-1], timesteps[1:]):
            trajectory[condition_mask] = condition_data[condition_mask]

            t = scheduler.timesteps_to_times(b)
            next_t = scheduler.timesteps_to_times(next_b)

            denoise = lambda traj, t: model(traj, t, local_cond=local_cond, global_cond=global_cond)
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(denoise, trajectory, t, next_t, clamp = True)

        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]      

        return trajectory

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # assert 'past_action' not in obs_dict # not implemented yet
        # normalize input

        ## torch.Size([32, 128, 4])
        nobs = self.normalizer['observations'].normalize(obs_dict['observations'])
        B = nobs.shape[0]
        # To = value.shape[0]
        T = self.horizon
        # Da = self.action_dim
        Do = self.obs_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            # this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            # nobs_features = this_nobs
            # reshape back to B, Do
            nobs_features = torch.cat((nobs[:,0,...],nobs[:,-1,...]), dim=1)
            global_cond = nobs_features.reshape(B, -1)
            
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = this_nobs
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,:] = nobs_features
            cond_mask[:,:To,:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond)
        
        # unnormalize prediction
        nstates_pred = nsample

        # start = To - 1
        # end = start + self.n_action_steps

        states_pred = self.normalizer['observations'].unnormalize(nstates_pred)

        # get action

        # action = action_pred[:,start:end]
        
        result = {
            'observations': states_pred[0],
            'observations_pred': states_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch

        '''
        (obs): Object of type: ParameterDict
        (action): Object of type: ParameterDict
        (reward): Object of type: ParameterDict
        '''
        nobs = self.normalizer['observations'].normalize(batch['observations'])
        # nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nobs.shape[0]
        # horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nobs
        # cond_data = torch.cat((nobs[:,0,...], nobs[:,-1,...]), dim=1).reshape(batch_size, -1)

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            # this_nobs = dict_apply(nobs, 
            #     lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            # reshape back to B, Do
            # nobs_features = nobs[:,:self.n_obs_steps,...]
            global_cond = torch.cat((nobs[:,0,...], nobs[:,-1,...]), dim=1).reshape(batch_size, -1)

            '''
            trajectory: 
            torch.Size([32, 128, 4])
            global cond: 
            torch.Size([32, 8])
            '''
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = this_nobs
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, self.horizon, -1)
            # cond_data = torch.cat([nactions, nobs_features], dim=-1)
            cond_data = nobs_features
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)


        # Sample a random timestep for each image
        times, _ = self.noise_scheduler.sample_times(trajectory)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, times)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        # noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the initial state
        denoise = lambda traj, t: self.model(traj, t, local_cond=local_cond, global_cond=global_cond)
        pred = self.noise_scheduler.calc_out(denoise, noisy_trajectory, times, clamp=False)
        weights = self.noise_scheduler.get_weights(times, None, "karras")
        
        target = trajectory


        loss = Huber_Loss(pred, target, delta = self.delta, weights = weights)
        return loss
