import os
import sys
import torch
from .ppo import PPO
from .transformer.transformerPPOagent import Agent, TransformerPPOAC
from .transformer.config import transformerconfig, ppoconfig

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
sys.path.insert(0, root_dir)

from utils.algo_utils import TerminationCondition
import gym
import evogym.envs
import numpy as np
import random

def run(env_name, robots, seed, pop_size, train_iters, ac_type=None, device_num=1):

    # Load configs
    trans_args = transformerconfig()
    ppo_args = ppoconfig()
    ppo_args.env_name=env_name
    ppo_args.seed = seed
    ppo_args.device_num = device_num
    ppo_args.cuda = torch.cuda.is_available()
    ppo_args.cuda_deterministic= torch.cuda.is_available()
    ppo_args.TIMESTEPS=2048 if len(robots) > 1 else 512
    ppo_args.MULTI=True if len(robots) > 1 else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(ppo_args.device_num))
    
    # Seed
    random.seed(ppo_args.seed)
    np.random.seed(ppo_args.seed)
    torch.manual_seed(ppo_args.seed)
    torch.cuda.manual_seed_all(ppo_args.seed)
    
    # Set GPU device
    if ppo_args.cuda and torch.cuda.is_available() and ppo_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    # Set dimensions
    train_env = gym.make(env_name, mode='modular', body=robots[0][0], connections=robots[0][1],env_id=env_name)
    modular_state_dim = train_env.modular_state_dim
    modular_action_dim = train_env.modular_action_dim
    other_feature_size = train_env.other_dim
    sequence_size = train_env.voxel_num
    train_env.close()
    
    # Dirs
    if ppo_args.MULTI:
        experiment_name = f'multi_{env_name}_seed_{ppo_args.seed}'
    else:
        experiment_name = f'{ac_type}_PPO_{env_name}_seed_{ppo_args.seed}'
   
    home_path = os.path.join(root_dir, "saved_data",env_name, experiment_name)
    try:
        os.makedirs(home_path)
    except:
        pass
    temp_path = os.path.join(root_dir, "saved_data", env_name, experiment_name, "metadata.txt") 

    # Save metadata
    f = open(temp_path, "w")
    f.write(f'ENV NAME: {env_name}\n')
    for i in range(len(robots)):f.write(f'ROBOT_BODY{i}: {robots[i][0]}\n')
    f.write(f'SEED: {ppo_args.seed}\n')
    f.write(f'POP SIZE: {pop_size}\n')
    f.write(f'DEVICE NUM: {ppo_args.device_num}\n')
    f.write(f'TRAIN ITERS: {train_iters}\n')
    f.write(f'AC TYPE: {ac_type}\n')
    f.write(f'USE POS EMBEDDING: {trans_args.POS_EMBEDDING}\n')
    f.write(f'ACT FIXED NOISE: {ppo_args.ACTION_STD_FIXED}\n')
    f.write(f'OBS ENCODER: {trans_args.use_other_obs_encoder}\n')
    f.write(f'CONDITION DECODER: {trans_args.condition_decoder}\n')
    f.close()

    # Termination condition  
    tc = TerminationCondition(train_iters)
    
    # Init PPO Actor-Critic
    actor_critic = TransformerPPOAC(modular_state_dim=modular_state_dim, modular_action_dim=modular_action_dim,
                                    sequence_size=sequence_size, other_feature_size=other_feature_size, 
                                    ppo_args=ppo_args,trans_args=trans_args, ac_type=ac_type, device=device)
    # Universal agent
    uni_agent = Agent(actor_critic=actor_critic)
    uni_agent.ac.to(device)

    # Training
    print("Parameters to be optimized: ")
    for name, param in uni_agent.named_parameters():
        print(name, param.requires_grad)
            
    # PPO
    ppo = PPO(robots=robots, termination_condition=tc,pop_size=pop_size,agent=uni_agent, verbose=True, 
                ppo_args=ppo_args, save_path=home_path)
    
    # Start training
    ppo.train()


if __name__ == "__main__":
    run()