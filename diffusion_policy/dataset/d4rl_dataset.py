from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizerNew, SingleFieldGaussianNormalizerNew, GaussianNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, NewSequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_range_normalizer_from_stat_new,
    get_gaussian_normalizer_from_stat_new,
    array_to_stats
)

import gym
import pdb
import collections
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
import os


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl


def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env.spec.max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()
    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)
    return dataset




def maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        xy = dataset['observations'][:,:2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        return dataset

    return _fn




def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_obs'] = next_observations
    return episode


def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:length]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened



def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )





class D4RLDataset(BaseLowdimDataset):
    def __init__(self,
            env_name: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
        ):

        env = load_environment(env_name)
        dataset = get_dataset(env)
        if 'maze2d' in env_name:
            dataset = maze2d_set_terminals(env)(dataset)
        self.max_episode_steps = env.max_episode_steps
        self.env_name = env_name
        
        print(dataset.keys())
        N = dataset['rewards'].shape[0]    ## 402000
        data_ = collections.defaultdict(list)

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = 'timeouts' in dataset

        replay_buffer = ReplayBuffer.create_empty_numpy()

        episode_step = 0
        traj_num = 0

        for i in range(N):
            ## bad termination
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts: ## yes
                ## good termination
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env.max_episode_steps - 1)

            for k in dataset:
                ### next_observations & observations
                if 'metadata' in k: continue
                data_[k].append(dataset[k][i])

            if done_bool or final_timestep:
                traj_num += 1
                episode_step = 0
                episode_data = {}
                for k in data_:
                    # assert done_bool == len(np.array(data_[k])) < env.max_episode_steps
                    if 'reward' in k: 
                        episode_data[k] = self._convert_reward(np.array(data_[k]))
                    else: 
                        episode_data[k] = np.array(data_[k])
                if 'maze2d' in env.name: 
                    episode_data = process_maze2d_episode(episode_data)

                ## for maze2d large ddim only
                if len(episode_data['actions'])>horizon:
                # print(len(episode_data['actions']))
                    replay_buffer.add_episode(episode_data)
                # yield episode_data
                data_ = collections.defaultdict(list)
            else:
                episode_step += 1

        
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set


    def get_normalizer(self, mode='limits', **kwargs):
        # data = {
        #     'observations': self.replay_buffer['observations'],
        #     'actions': self.replay_buffer['actions']
        # }
        # if 'range_eps' not in kwargs:
        #     # to prevent blowing up dims that barely change
        #     kwargs['range_eps'] = 5e-2
        # normalizer = LinearNormalizer()
        # normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        normalizer = LinearNormalizer()
        ## action shape: (400058, 3)
        stat = array_to_stats(self.replay_buffer['actions'])
        this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer['actions'] = this_normalizer

        stat = array_to_stats(self.replay_buffer['observations'])
        this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer['observations'] = this_normalizer

        stat = array_to_stats(self.replay_buffer['rewards'])
        this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer['rewards'] = this_normalizer
        return normalizer




    def _convert_reward(self, raw_reward, discount=0.99, returns_scale=1.0, termination_penalty=0.0):
        '''
        raw_rewards shape: (traj_len, )
        maximum q value: ??? if we want to normalize
        '''
        if not 'maze2d' in self.env_name: 
            if len(raw_reward) < self.max_episode_steps:
                raw_reward[-1] += termination_penalty
        discounts = discount ** np.arange(raw_reward.shape[0])
        q_values = np.array([np.sum(discounts[:(raw_reward.shape[0]-index)] * raw_reward[index:]) for index in range(raw_reward.shape[0])], dtype=np.float32)
        q_values = q_values/returns_scale
        
        # q_values = np.array([q_values/returns_scale], dtype=np.float32)
        # print(q_values)
        # print(q_values.mean())
        # print(q_values.std())
        return q_values[:, None]



    def _sample_to_data(self, sample):

        '''
        obs
        action
        reward
        terminals
        '''
        data = {
            'observations': sample['observations'],
            'actions': sample['actions'], # T, D_a
            'rewards': sample['rewards'], 
        }
        return data
    

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['actions'])
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
