import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.mujoco_rendering import MuJoCoRenderer, Maze2dRenderer
from gym.wrappers import FlattenObservation
import gym
import os
import time


from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

from diffusion_policy.gym_util.mujoco_video import save_video



class D4RLLowdimKitchenRunner(BaseLowdimRunner):
    def __init__(self,
            env_name, 
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=5,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        print(env_name)
        wrapped_env = gym.make(env_name)
        env = wrapped_env.unwrapped
        env.max_episode_steps = wrapped_env._max_episode_steps
        env.name = env_name
        self.env_name = env_name


        env_seeds_train = list()
        env_seeds_test = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            env_seeds_train.append(seed)

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            env_seeds_test.append(seed)


        task_fps = 10
        steps_per_render = max(10 // fps, 1)

        env = MultiStepWrapper(
                env, 
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        self.env = env
        self.n_envs = n_envs
        self.n_train = n_train
        self.n_test = n_test
        self.env_seeds_train = env_seeds_train
        self.env_seeds_test = env_seeds_test
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        if 'maze' in env_name:
            self.renderer = Maze2dRenderer(env_name)
        else: 
            self.renderer = MuJoCoRenderer(env)

    


    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = self.n_envs
        # n_inits = len(self.env_init_fn_dills)
        # n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        # all_video_paths = [None] * n_inits
        all_rewards_train = []
        all_scores_train = []
        all_rewards_test = [] 
        all_scores_test = []
        last_info_train = [] 
        last_info_test = [] 

        all_times_train = []
        all_times_test = []

        for chunk_idx in range(self.n_train):
            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            total_reward = []
            times = []
            observations = []
            observations.append(obs[-1,:])
            # total_info = []

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval D4RL train {chunk_idx+1}/{self.n_train}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    'observations': obs.astype(np.float32)
                }
                # np_obs_dict = {
                #     'obs': obs.astype(np.float32)
                # }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                start_time = time.time()

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                t = time.time() - start_time
                times.append(t)
                # print(t)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                # obs shape: (2, 11)
                total_reward.append(reward) 
                observations.append(obs[-1,:])
                # total_info.append(info)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[0])
            pbar.close()

            print('total reward: ')
            print(np.sum(total_reward))
            print('total score: ')
            print(self.env.get_normalized_score(np.sum(total_reward)))
            all_rewards_train.append(total_reward)
            all_times_train.append(times)
            last_info_train.append(info)
            # render video for this round
            observations = np.array(observations)[None, :]
            if 'maze' in self.env_name:
                savepath = self.output_dir+'/train'+str(chunk_idx)+".png"
                self.renderer.composite(savepath, observations)
            else: 
                self.show_sample(self.renderer, observations, filename='train' + str(chunk_idx) + ".mp4", savebase=self.output_dir)


        for chunk_idx in range(self.n_test):
            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            total_reward = []
            times = []
            observations = []
            observations.append(obs[-1,:])
            # total_info = []

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval D4RL test {chunk_idx+1}/{self.n_test}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    'observations': obs.astype(np.float32)
                }
                # np_obs_dict = {
                #     'obs': obs.astype(np.float32)
                # }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy

                start_time = time.time()

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                t = time.time() - start_time
                times.append(t)
                # print(t)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                total_reward.append(reward)
                observations.append(obs[-1,:])
                # total_info.append(info)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[0])
            pbar.close()

            observations = np.array(observations)[None, :]
            if 'maze' in self.env_name:
                savepath = self.output_dir+'/test'+str(chunk_idx)+".png"
                self.renderer.composite(savepath, observations)
            else: 
                self.show_sample(self.renderer, observations, filename='test' + str(chunk_idx) + ".mp4", savebase=self.output_dir)

            print('total reward: ')
            print(np.sum(total_reward))
            print('total scsore: ')
            print(self.env.get_normalized_score(np.sum(total_reward)))
            all_rewards_test.append(total_reward)
            all_times_test.append(times)
            last_info_test.append(info)
            # collect data for this round
            # all_video_paths[this_global_slice] = env.render()[this_local_slice]
            # all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            # last_info[this_global_slice] = [dict((k,v[-1]) for k, v in x.items()) for x in info][this_local_slice]

        # log
        total_rewards = collections.defaultdict(list)
        total_scores = collections.defaultdict(list)
        total_times = collections.defaultdict(list)
        # max_rewards = collections.defaultdict(list)
        # total_p1 = collections.defaultdict(list)
        # total_p2 = collections.defaultdict(list)
        prefix_event_counts = collections.defaultdict(lambda :collections.defaultdict(lambda : 0))
        prefix_counts = collections.defaultdict(lambda : 0)

        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(self.n_train):
            seed = self.env_seeds_train[i]
            prefix = 'train/'
            this_rewards = all_rewards_train[i]
            total_reward = np.array(this_rewards).sum() # (0, 0.49, 0.51)
            mean_time = np.array(all_times_train[i]).mean()
            # max_reward = np.max(all_rewards_train[i])

            total_rewards[prefix].append(total_reward)
            total_scores[prefix].append(self.env.get_normalized_score(total_reward))
            total_times[prefix].append(mean_time)
            # total_p1[prefix].append(p1)
            # total_p2[prefix].append(p2)
            log_data[prefix+f'sim_max_reward_{seed}'] = total_reward

            # aggregate event counts
            # prefix_counts[prefix] += 1
            # for key, value in last_info_train[i].items():
            #     delta_count = 1 if value > 0 else 0
            #     prefix_event_counts[prefix][key] += delta_count


        
        for i in range(self.n_test):
            seed = self.env_seeds_test[i]
            prefix = 'test/'
            this_rewards = all_rewards_test[i]
            total_reward = np.array(this_rewards).sum() # (0, 0.49, 0.51)
            mean_time = np.array(all_times_test[i]).mean()
            
            total_rewards[prefix].append(total_reward)
            total_scores[prefix].append(self.env.get_normalized_score(total_reward))
            total_times[prefix].append(mean_time)
            # total_p1[prefix].append(p1)
            # total_p2[prefix].append(p2)
            log_data[prefix+f'sim_max_reward_{seed}'] = total_reward

            # aggregate event counts
            # prefix_counts[prefix] += 1
            # for key, value in last_info_test[i].items():
            #     delta_count = 1 if value > 0 else 0
            #     prefix_event_counts[prefix][key] += delta_count


        # log aggregate metrics
        for prefix, value in total_rewards.items():
            name = prefix+'mean_reward'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in total_scores.items():
            name = prefix+'mean_scores'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in total_scores.items():
            name = prefix+'std_scores'
            value = np.std(value)
            log_data[name] = value
        
        for prefix, value in total_times.items():
            name = prefix+'mean_times'
            value = np.mean(value)
            log_data[name] = value
            
        # for prefix, value in total_p1.items():
        #     name = prefix+'p1'
        #     value = np.mean(value)
        #     log_data[name] = value
        # for prefix, value in total_p2.items():
        #     name = prefix+'p2'
        #     value = np.mean(value)
        #     log_data[name] = value
        
        # summarize probabilities
        # for prefix, events in prefix_event_counts.items():
        #     prefix_count = prefix_counts[prefix]
        #     for event, count in events.items():
        #         prob = count / prefix_count
        #         key = prefix + event
        #         log_data[key] = prob

        print(log_data)
        return log_data


    # def show_diffusion(self, subsampled, n_repeat=10, savepath = None):
    #     images = []
    #     for t in tqdm(range(len(subsampled))):
    #         observation = subsampled[t]

    #         img = self.renderer.composite(None, observation)
    #         images.append(img)
    #     images = np.stack(images, axis=0)

    #     ## pause at the end of video
    #     images = np.concatenate([
    #         images,
    #         images[-1:].repeat(n_repeat, axis=0)
    #     ], axis=0)

    #     save_video(savepath, images)


    def show_sample(self, renderer, observations, filename='sample.mp4', savebase='/content/videos'):
        '''
            observations : [ batch_size x horizon x observation_dim ]
        '''

        savepath = os.path.join(savebase, filename)
        images = []
        for rollout in observations:
            ## [ horizon x height x width x channels ]
            img = renderer._renders(rollout, partial=True)
            images.append(img)

        ## [ horizon x height x (batch_size * width) x channels ]
        images = np.concatenate(images, axis=2)
        
        save_video(savepath, images, fps=self.fps)