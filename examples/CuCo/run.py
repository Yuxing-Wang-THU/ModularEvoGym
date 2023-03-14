import os
import sys
import torch
from .ppo import PPO
from .transformer.transformerPPOagent import Agent, TransformerPPOAC
from .transformer.config import transformerconfig, ppoconfig, ncaconfig
from .NCA import NCA
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
sys.path.insert(0, root_dir)
import numpy as np
import random
from evogym import sample_robot
from utils.algo_utils import TerminationCondition
import gym
import evogym.envs
import copy

def run(env_name, seed, target_design_size, pop_size, train_iters, ac_type=None, device_num=1,rl_only=None):

    # Load configs
    trans_args = transformerconfig()
    ppo_args = ppoconfig()
    nca_args = ncaconfig()
    ppo_args.env_name=env_name
    ppo_args.seed = seed
    ppo_args.device_num = device_num
    ppo_args.cuda = torch.cuda.is_available()
    ppo_args.cuda_deterministic= torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ppo_args.device_num is not None:
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

    # Pre-defined Curriculum [3x3, 5x5, 7x7]
    stages = np.arange(3, target_design_size+1, 2)

    # No curriculum, only RL
    if rl_only:
        stages = [target_design_size]

    # Set dimensions
    body, connection = sample_robot((5,5))
    train_env = gym.make(env_name,mode='modular', body=body, connections=connection,env_id=env_name)
    modular_state_dim = train_env.modular_state_dim
    modular_action_dim = train_env.modular_action_dim
    other_feature_size = train_env.other_dim
    obs_sample = train_env.get_modular_obs()
    train_env.close()
    
    if env_name == 'Jumper-v0':
        ppo_args.ACTION_STD_FIXED=False
        trans_args.use_other_obs_encoder=True
    
    if env_name == 'Thrower-v0':
        trans_args.use_other_obs_encoder=True

    # Dirs
    experiment_name = f'CuCo_{env_name}_seed_{ppo_args.seed}'
    if rl_only:
        experiment_name = f'CuCo-NCU_{env_name}_seed_{ppo_args.seed}'

    home_path = os.path.join(root_dir, "saved_data",env_name, experiment_name)
    try:
        os.makedirs(home_path)
    except:
        pass
    temp_path = os.path.join(home_path, "metadata.txt") 

    # Save metadata
    f = open(temp_path, "w")
    f.write(f'ENV NAME: {env_name}\n')
    f.write(f'SEED: {ppo_args.seed}\n')
    f.write(f'POP SIZE: {pop_size}\n')
    f.write(f'DEVICE NUM: {ppo_args.device_num}\n')
    f.write(f'TRAIN ITERS: {train_iters}\n')
    f.write(f'AC TYPE: {ac_type}\n')
    f.write(f'USE POS EMBEDDING: {trans_args.POS_EMBEDDING}\n')
    f.write(f'ACT FIXED NOISE: {ppo_args.ACTION_STD_FIXED}\n')
    f.write(f'OBS ENCODER: {trans_args.use_other_obs_encoder}\n')
    f.write(f'CONDITION DECODER: {trans_args.condition_decoder}\n')
    f.write(f'RL ONLY: {rl_only}\n')
    f.close()

    # Termination condition  
    tc = TerminationCondition(train_iters)
    
    # Init PPO Actor-Critic
    actor_critic = TransformerPPOAC(modular_state_dim=modular_state_dim, modular_action_dim=modular_action_dim,
                                    sequence_size=int(stages[0]*stages[0]), other_feature_size=other_feature_size, 
                                    ppo_args=ppo_args,trans_args=trans_args, ac_type=ac_type)
    # Neural Cellular Automata
    nca = NCA(settings=nca_args, obs_sample=obs_sample, device=device)
    
    # Universal agent
    uni_agent = Agent(actor_critic=actor_critic,nca=nca)

    # Training
    for stage in stages:
        init_designs = []
        ### MAKE STAGE DIRECTORIES ###
        save_path_stage = os.path.join(home_path, "stage_" + str(stage))
        try:
            os.makedirs(save_path_stage)
        except:
            pass

        # Inherit
        nca_args.im_size = stage + 2
        nca_args.learn_stage = stage
        ppo_args.DESIGN_SIZE = int(stage * stage)
        new_nca = NCA(settings=nca_args, obs_sample=obs_sample, device=device)

        new_nca.designer = copy.deepcopy(uni_agent.nca.designer)
        
        uni_agent.nca = new_nca
        uni_agent.ac.reset_seq_size(int(stage*stage))
        uni_agent.ac.to(device)
        uni_agent.nca.designer.to(device)
        
        print("Parameters to be optimized: ")
        for name, param in uni_agent.named_parameters():
            print(name, param.requires_grad)

        if stage == stages[0]:
            for i in range(pop_size):  
                init_designs.append(uni_agent.nca.get_init_design())
        else:
            for i in range(pop_size):  
                new_design = uni_agent.nca.get_init_design()
                if returned_robot is not None:
                    new_design[0][2:stage, 2:stage] = copy.deepcopy(returned_robot.body) 
                
                if env_name == 'Jumper-v0':
                    new_design = uni_agent.nca.get_init_design()
                init_designs.append(new_design)
                
        # Employ ppo
        ppo = PPO(stage=stage,termination_condition=tc,init_designs=init_designs, pop_size=pop_size,agent=uni_agent, verbose=True, 
                  ppo_args=ppo_args, nca_args=nca_args, save_path_stage=save_path_stage)

        returned_robot = ppo.train()

if __name__ == "__main__":
    run()