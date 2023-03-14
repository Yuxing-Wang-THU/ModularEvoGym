from CuCo.run import run
import argparse

if __name__ == "__main__":
     # Parser
    parser = argparse.ArgumentParser(description='PyTorch args')
    parser.add_argument('--env', type=str, default='Walker-v0',
                        help='env')
    parser.add_argument('--seed', type=int, default=101,
                        help='random seed')
    parser.add_argument('--ac_type', type=str, default="transformer",
                        help='(transformer or fc)')
    parser.add_argument('--target_size', type=int, default=7,
                        help='target design space, default 7x7') 
    parser.add_argument('--rl_only', action='store_true',
                        help='end-to-end rl-based co-design, CuCo-NCU') 
    parser.add_argument('--device_num', type=int, default=0,
                        help='gpu device id')
    parser.add_argument('--threads_num', type=int, default=4,
                        help='number of CPU threads')
    parser.add_argument('--train_iters', type=int, default=1000,
                        help='policy iterations per stage')                                        
    args = parser.parse_args()

    run(env_name=args.env, target_design_size=args.target_size, seed=args.seed, pop_size=args.threads_num,
        train_iters=args.train_iters, ac_type=args.ac_type, device_num=args.device_num,rl_only=args.rl_only)