from transformer_ppo.run import run
import argparse
from evogym import sample_robot

# 5 robots to be controlled
robot_num = 5
robots = [sample_robot((5,5)) for _ in range(robot_num)]

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch Args')
    parser.add_argument('--env', type=str, default='Walker-v0',
                        help='random seed')
    parser.add_argument('--seed', type=int, default=101,
                        help='random seed')
    parser.add_argument('--ac_type', type=str, default="transformer",
                        help='(transformer, fc)')
    parser.add_argument('--device_num', type=int, default=0,
                        help='gpu device id')
    parser.add_argument('--train_iters', type=int, default=3000,
                        help='policy iterations')                             
    args = parser.parse_args()

    # Run
    run(env_name=args.env, robots=robots, seed=args.seed, pop_size=robot_num,
        train_iters=args.train_iters, ac_type=args.ac_type, device_num=args.device_num)