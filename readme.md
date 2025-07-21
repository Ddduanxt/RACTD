# Reward Aware Consistency Trajectory Distillation

The official PyTorch implementation of [[Accelerating Diffusion Models in Offline RL via Reward-Aware Consistency Trajectory Distillation]](https://arxiv.org/pdf/2506.07822). 





## Overview
RACTD is a one step diffusion model for offline RL that is distilled from a multi-step diffusion teacher model and a standalone reward model. More details can be found in [our paper](https://arxiv.org/pdf/2506.07822). 

## Installation
The below instructions are copied from [Consistency Policy](https://github.com/Aaditya-Prasad/consistency-policy/). 

Install our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 20.04 you need to install the following apt packages for mujoco:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Create a conda environment: 

```console
$ conda env create -f environment.yaml
```

You can also login to wandb if you wish to log your training runs. 
```console
conda activate ractd
wandb login
```

## Sampling with pretrained weights

Pretrained weights of the teacher, reward, and student models can be found [here](https://drive.google.com/drive/folders/1xQSbByA_Fhgf5NvIxlIBMBpYWsgoFmG6?usp=drive_link). To download all checkpoints, run

```console
pip install gdown==4.6.1
gdown --folder https://drive.google.com/drive/folders/1xQSbByA_Fhgf5NvIxlIBMBpYWsgoFmG6?usp=drive_link
```

If you just want to run sampling for a specific environment, download the weights of teacher (edm), reward (reward), and guided_student (guided_ctmp) of that environment. 


To run sampling in the MuJoCo (e.g. hopper) environment, substitute ```policy.teacher_path & policy.edm```, ```policy.reward_path```, ```training.load_path```, and ```training.output_dir``` to path to the weight of teacher, reward model,  student model, and output directory respectively. Change ```task.env_runner.env_name``` to the corresponding environment. 

```console
python train.py --config-dir=configs/ --config-name=rollout_student_d4rl_hopper.yaml logging.name=<wandb_logging_name> task.env_runner.env_name=hopper-medium-replay-v2
```

## Training
Training is done similarly to [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [Consistency Policy](https://github.com/Aaditya-Prasad/consistency-policy/). The user defines a config yaml file with their desired parameters. We need to train a teacher model and a reward model first, then use both models to train a RACTD student. 



### Teacher Network (EDM)

We follow the same setup in [Consistency Policy](https://github.com/Aaditya-Prasad/consistency-policy/) to train an EDM teacher model. Specify ```training.output_dir``` to the output directory where you want to store the model weights. Here is an example to train a teacher model for hopper-medium-replay: 


```console
(consistency-policy)[consistency-policy]$ python train.py --config-dir=configs/ --config-name=edm_d4rl_hopper_mr.yaml logging.name=teacher_d4rl_hopper_mr
```


### Reward Model 

To train a reward model, specify ```training.output_dir``` and run: 

```console
(consistency-policy)[consistency-policy]$ python train.py --config-dir=configs/ --config-name=reward_d4rl_hopper_mr.yaml logging.name=reward_d4rl_hopper_mr
```


### Reward guided student distillation 
Once you have a trained teacher and reward checkpoint, you can distill a RACTD model. Set ```policy.teacher_path``` and ```policy.edm``` (for warm start) in the config to the path to teacher weight. Set ```policy.reward_path``` to the path to reward weight. Set ```training.output_dir``` to the path to the output directory. 


```console
(consistency-policy)[consistency-policy]$ python train.py --config-dir=configs/ --config-name=guided_ctmp_hopper_mr.yaml logging.name=guided_student_hopper_mr
```


## Citation
If you find our work interesting or helpful, please consider citing
```


```


## Acknowledgement
Our implementation is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [Consistency Policy](https://github.com/Aaditya-Prasad/consistency-policy/), and [Diffuser](https://github.com/jannerm/diffuser)