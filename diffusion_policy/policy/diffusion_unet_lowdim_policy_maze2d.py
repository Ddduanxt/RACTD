from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
# from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from consistency_policy.diffusion_unet_with_dropout import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionUnetLowdimMazePolicy(BaseLowdimPolicy):
    def __init__(self, 
            # model: ConditionalUnet1D,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=32,
            down_dims=(64,128,256),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond


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
        print('noise scheduler: ')
        print(self.noise_scheduler)
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
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        nobs = self.normalizer['observations'].normalize(obs_dict['observations'])
        # B, _, Do = nobs.shape
        B = nobs.shape[0]
        Do = nobs.shape[-1]
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through local feature
            # all zero except first To timesteps
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
            global_cond=global_cond,)
        
        # unnormalize prediction
        nstates_pred = nsample
        states_pred = self.normalizer['observations'].unnormalize(nstates_pred)

        # get action
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
        # nbatch = self.normalizer.normalize(batch)
        # obs = nbatch['obs']
        # action = nbatch['action']

        nobs = self.normalizer['observations'].normalize(batch['observations'])
        batch_size = nobs.shape[0]
        # action = self.normalizer['actions'].normalize(batch['actions'])

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nobs
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

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
