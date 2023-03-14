import numpy as np
import torch
from . import helper
from .envs import make_vec_envs

def evaluate(
    num_evals, 
    uni_agent, 
    ob_rms,
    env_name, 
    init_robots,
    seed=1, 
    device=None):

    eval_envs = make_vec_envs(env_name, init_robots, seed, None, device, ret=False, ob=True)
    vec_norm = helper.get_vec_normalize(eval_envs)
    
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    # recorders
    eval_episode_rewards = []

    obs = eval_envs.reset()
    while True:
        with torch.no_grad():
            val, action, logp = uni_agent.uni_act(obs, mean_action=True)
        obs, rewards, done, infos = eval_envs.step(action)
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
        # check
        if num_evals*len(init_robots) == len(eval_episode_rewards):
            break
    eval_envs.close()
    print("Evalution done!")
    return np.average(eval_episode_rewards)

   