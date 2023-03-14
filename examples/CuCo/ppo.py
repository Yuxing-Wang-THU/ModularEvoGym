import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from . import helper
from .evaluate import evaluate
from .envs import make_vec_envs
import numpy as np
from collections import deque
from utils.algo_utils import Structure
from .transformer.buffer import Buffer
import csv
import gym
import evogym.envs
from evogym import sample_robot
import copy
# PPO class
class PPO:
    def __init__(self, stage=None,init_designs=None, termination_condition=None,pop_size=None, agent=None,
    verbose=True, ppo_args=None,nca_args=None,save_path_stage=None) -> None:
        self.args = ppo_args
        self.verbose = verbose
        self.agent = agent
        self.init_designs = init_designs
        self.designs = copy.deepcopy(init_designs)
        self.robots = []
        self.termination_condition = termination_condition
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.lr, eps=self.args.EPS)
        self.pop_size = pop_size
        self.stage = stage
        self.total_episode = 0
        self.nca_args = nca_args
        self.save_path_stage = save_path_stage
        self.eval_threads = 2

        for i in range(self.pop_size):  
            self.robots.append(Structure(*sample_robot((self.stage, self.stage)), i))

        ### MAKE eval robots ###
        self.eval_robots = [self.robots[0]] * self.eval_threads
        self.fitness_log = open(self.save_path_stage + f"/fitness_log{self.stage}.csv", 'w')
        self.fitness_logger = csv.DictWriter(self.fitness_log, fieldnames=("iters","stage", "timesteps",'total_episode',"average_score",'train_score'))
        self.save_path_structures = os.path.join(self.save_path_stage, "structures")
        self.save_path_controllers = os.path.join(self.save_path_stage, "controllers")

        try:
            os.makedirs(self.save_path_structures)
        except:
            pass
        try:
            os.makedirs(self.save_path_controllers)
        except:
            pass

        self.buffer = self.reset_buffer()
        self.buffer.to(self.device)


    def reset_buffer(self):
        body, connection = sample_robot((self.stage, self.stage))
        train_env = gym.make(self.args.env_name,mode='modular', body=body, 
                                connections=connection,env_id=self.args.env_name)
        sequence_size = train_env.voxel_num
        obs_sample = train_env.reset()
        train_env.close()
        return Buffer(obs_sample, act_shape=sequence_size, num_envs=self.pop_size, cfg=self.args)

    def train(self):
        # generate designs
        print("INIT designs: ", self.init_designs)
        # # Not normalize env
        self.envs = make_vec_envs(self.args.env_name, self.robots, self.args.seed, 
                                self.args.GAMMA, self.device, ob=True, ret=True,
                                nca_setting=self.nca_args, init_nca_design=self.init_designs[0])

        start = time.time()
        obs = self.envs.reset()
        num_updates = int(self.args.num_env_steps) // self.args.TIMESTEPS // self.pop_size
        episode_rewards = deque(maxlen=10)

        for j in range(num_updates*4):
            if self.args.use_linear_lr_decay:
                # decrease learning rate linearly
                helper.update_linear_schedule(self.optimizer, j, num_updates*4,
                     self.args.lr) 
            print(" Collect experience !!!!!")

            for step in range(self.args.TIMESTEPS):
                val, act, logp = self.agent.uni_act(obs)
                next_obs, reward, done, infos = self.envs.step(act)
                # If done then clean the history of observations.
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                for done_ in done:
                    if done_:
                        self.total_episode +=1

                masks = torch.tensor(
                        [[0.0] if done_ else [1.0] for done_ in done],
                        dtype=torch.float32,
                        device=self.device,
                    )
                timeouts = torch.tensor(
                        [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos],
                        dtype=torch.float32,
                        device=self.device,
                    )

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts)
                obs = next_obs

            next_val = self.agent.get_value(obs)
            self.buffer.compute_returns(next_val)

            self.train_on_batch()
            
            # evaluate mean structure
            if (self.args.eval_interval is not None and j % self.args.eval_interval == 0):
                print("Begin Evaluation!!!!!")
                init_test_design = copy.deepcopy(self.init_designs[0]) 
                total_num_steps = (j + 1) * self.pop_size * self.args.TIMESTEPS
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {} \n".format(j, total_num_steps,
                            int(total_num_steps / (end - start)),))

                obs_rmss = helper.get_vec_normalize(self.envs).ob_rms

                avg_reward, designed_robot = evaluate(num_evals=self.args.num_evals, uni_agent=self.agent,ob_rms=obs_rmss, env_name=self.args.env_name,
                                            init_robots=self.eval_robots,init_nca_design=init_test_design,
                                            seed = self.args.seed, nca_setting=self.nca_args, device=self.device)
                
                res = {"iters": j, "stage":self.stage,"timesteps": total_num_steps,'total_episode':self.total_episode, "average_score":avg_reward
                        ,'train_score':np.average(episode_rewards)}
                
                self.fitness_logger.writerow(res)
                self.fitness_log.flush()
                
                ### SAVE EVAL STRUCTURE ###
                if designed_robot is not None:
                    temp_path_body = os.path.join(self.save_path_structures, f'iter_{str(j)}')
                    np.savez(temp_path_body, designed_robot.body, designed_robot.connections)
                    temp_path_controller = os.path.join(self.save_path_controllers, f'iter_{str(j)}' + ".pt")
                    print("SAVE OBS_RMS!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(getattr(helper.get_vec_normalize(self.envs), 'ob_rms', None))
                    torch.save([self.agent, getattr(helper.get_vec_normalize(self.envs), 'ob_rms', None)], temp_path_controller)
                
                ### SAVE EVAL CONTROLLER ###
                if self.verbose:
                    print(f'Current Mean reward: {avg_reward}\n')
            
            # return upon reaching the termination condition
            if not self.termination_condition == None:
                if self.termination_condition(j):
                    self.envs.close()
                    if designed_robot is not None:
                        print("Returned robot: ", designed_robot)
                        return_robot = copy.deepcopy(designed_robot)
                    else:
                        return_robot = None
                    if self.verbose:
                        print(f'Met termination condition ({j})...terminating...\n')
                    return return_robot
    
    def train_on_batch(self):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        for ep in range(self.args.EPOCHS):
            batch_sampler = self.buffer.get_sampler(adv)
            for batch in batch_sampler:
                # Reshape to do in a single forward pass for all steps
                val, logp, ent = self.agent(batch["obs"], batch["act"])
                clip_ratio = self.args.CLIP_EPS
                ratio = torch.exp(logp - batch["logp_old"])

                surr1 = ratio * batch["adv"]
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)*batch["adv"]
                
                pi_loss = -torch.min(surr1, surr2).mean()

                if self.args.USE_CLIP_VALUE_FUNC:
                    val_pred_clip = batch["val"] + (val - batch["val"]).clamp(
                        -clip_ratio, clip_ratio
                    )
                    val_loss = (val - batch["ret"]).pow(2)
                    val_loss_clip = (val_pred_clip - batch["ret"]).pow(2)
                    val_loss = 0.5 * torch.max(val_loss, val_loss_clip).mean()
                else:
                    val_loss = 0.5 * (batch["ret"] - val).pow(2).mean()

                self.optimizer.zero_grad()
                loss = val_loss * self.args.VALUE_COEF + pi_loss - ent * self.args.ENTROPY_COEF
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                value_loss_epoch += val_loss.item()
                action_loss_epoch += pi_loss.item()
                dist_entropy_epoch += ent.item()

            print("ratio: ",ratio.mean().item(), "ADV:",batch["adv"].mean().item())

        num_updates = self.args.EPOCHS * self.args.NUM_MINI_BATCH_SIZE

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        print('---------------------------------')
        print("PI loss: ",action_loss_epoch)
        print("value loss: ",value_loss_epoch)
        print("entropy: ", dist_entropy_epoch)
        print('---------------------------------')