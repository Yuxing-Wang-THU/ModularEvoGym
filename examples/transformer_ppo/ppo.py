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
from modular_envs.wrappers.modular_wrapper import modular_env

# PPO class
class PPO:
    def __init__(self, robots=None, termination_condition=None,pop_size=None, agent=None,
    verbose=True, ppo_args=None, save_path=None):
        self.args = ppo_args
        self.verbose = verbose
        self.agent = agent
        self.pop_size = pop_size
        self.robots = []
        self.robots_tuple = robots
        # For multi robots
        if self.args.MULTI:
            # N threads, N robots
            for i in range(self.pop_size):  
                self.robots.append(Structure(*robots[i], i))
        # For a single robot
        else:
            # one robot, multi threads
            for i in range(self.pop_size):  
                self.robots.append(Structure(*robots[0], i))

        self.termination_condition = termination_condition
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.lr, eps=self.args.EPS)
        self.total_episode = 0
        
        print("Num params: {}".format(helper.num_params(self.agent.ac)))

        # Logger
        self.save_path_ = save_path
        self.header = ("iters", "timesteps",'total_episode',"eval_score",'train_score')
        # Record every robot's fitness
        if self.args.MULTI:
            for i in range(self.pop_size):
                self.header = self.header + (f"Robot_{i}_score",)

        self.fitness_log = open(self.save_path_ + f"/fitness_log.csv", 'w')
        self.fitness_logger = csv.DictWriter(self.fitness_log, fieldnames=self.header)
        self.save_path_controllers = os.path.join(self.save_path_, "controllers")
        try:
            os.makedirs(self.save_path_controllers)
        except:
            pass

        # Seed
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        
        # Set GPU device
        if self.args.cuda and torch.cuda.is_available() and self.args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)

        # Set replay buffer
        self.buffer = self.reset_buffer()
        self.buffer.to(self.device)

    def reset_buffer(self):
        train_env = gym.make(self.args.env_name, body=self.robots_tuple[0][0], connections=self.robots_tuple[0][1])
        train_env = modular_env(env=train_env, body=self.robots_tuple[0][0])
        sequence_size = train_env.voxel_num
        obs_sample = train_env.reset()
        train_env.close()
        return Buffer(obs_sample, act_shape=sequence_size, num_envs=self.pop_size, cfg=self.args)

    def train(self):
        # Make envs
        self.envs = make_vec_envs(self.args.env_name, self.robots, self.args.seed, 
                                self.args.GAMMA, self.device, ob=True, ret=True)
        start = time.time()
        obs = self.envs.reset()
        num_updates = int(self.args.num_env_steps) // self.args.TIMESTEPS // self.pop_size
        episode_rewards = deque(maxlen=10)

        for j in range(num_updates*4):
            if self.args.use_linear_lr_decay:
                # decrease learning rate linearly
                helper.update_linear_schedule(self.optimizer, j, num_updates*4,
                     self.args.lr) 

            print("Collect experience !!!!!")
            for step in range(self.args.TIMESTEPS):
                # Sample actions
                val, act, logp = self.agent.uni_act(obs)
                next_obs, reward, done, infos = self.envs.step(act)
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

            print("Begin training!!!!!")
            self.train_on_batch()
            
            # Evaluation 
            if (self.args.eval_interval is not None and j % self.args.eval_interval == 0):
                total_num_steps = (j + 1) * self.pop_size * self.args.TIMESTEPS
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {} \n".format(j, total_num_steps,
                            int(total_num_steps / (end - start)),))

                print("Begin evaluation!!!!!")
                obs_rmss = helper.get_vec_normalize(self.envs).ob_rms
                if self.args.MULTI:
                    population_score = []
                    res = {"iters": j,"timesteps": total_num_steps,'total_episode':self.total_episode,
                          'train_score':np.average(episode_rewards)}
                    for i in range(self.pop_size):
                        reward = evaluate(num_evals=self.args.num_evals, uni_agent=self.agent,ob_rms=obs_rmss, env_name=self.args.env_name,
                                            init_robots=[self.robots[i]]*2,seed=self.args.seed, device=self.device)
                        res[f"Robot_{i}_score"] = reward
                        population_score.append(reward)
                    res['eval_score'] = np.average(population_score)
                    avg_reward = np.average(population_score)
                else:
                    avg_reward = evaluate(num_evals=self.args.num_evals, uni_agent=self.agent,ob_rms=obs_rmss, env_name=self.args.env_name,
                                            init_robots=[self.robots[0]]*2,seed=self.args.seed, device=self.device)
                    res = {"iters": j,"timesteps": total_num_steps,'total_episode':self.total_episode, "eval_score":avg_reward
                        ,'train_score':np.average(episode_rewards)}

                self.fitness_logger.writerow(res)
                self.fitness_log.flush()
                
                # Save controller
                temp_path_controller = os.path.join(self.save_path_controllers, f'iter_{str(j)}' + ".pt")
                torch.save([self.agent, getattr(helper.get_vec_normalize(self.envs), 'ob_rms', None)], temp_path_controller)
               
                if self.verbose:
                    print(f'Mean reward: {avg_reward}\n')
            
            # return
            if not self.termination_condition == None:
                if self.termination_condition(j):
                    self.envs.close()
                    if self.verbose:
                        print(f'Met termination condition ({j})...terminating...\n')
    
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