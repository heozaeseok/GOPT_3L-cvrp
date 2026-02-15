import os
import sys
import torch
import gymnasium as gym
import numpy as np
import time

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path) 
sys.path.append(parent_path) 

from tianshou.utils.net.common import ActorCritic
from tianshou.data import Batch
from ts_train import build_net
import arguments
from tools import *
from masked_ppo import MaskedPPOPolicy
from cvrp_utils import CVRPParser 

def test_tf(args):
    # Device setup
    device = torch.device(f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # CVRP Data Load
    cvrp_file = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\3L_CVRP\3l_cvrp01.txt"
    if not os.path.exists(cvrp_file):
        print(f"Error: Data file not found at {cvrp_file}")
        exit()
        
    parser = CVRPParser(cvrp_file)
    veh_info = parser.vehicle_info
    args.env.container_size = (veh_info['length'], veh_info['width'], veh_info['height'])

    # Environment Setup
    test_env = gym.make(
        args.env.id, 
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type="cvrp",
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=False, 
        cvrp_parser=parser
    )

    # Model Setup
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr)
    
    policy = MaskedPPOPolicy(
        actor=actor, critic=critic, optim=optim,
        dist_fn=CategoricalMasked,
        action_space=test_env.action_space
    )
    policy.eval()
    
    # Load Weights
    if os.path.exists(args.ckp):
        policy.load_state_dict(torch.load(args.ckp, map_location=device))
        print(f"Successfully loaded: {args.ckp}")
    else:
        print(f"Checkpoint not found: {args.ckp}")
        exit()

    print(f"\nStart T/F Testing for {args.test_episode} episodes...")
    print(f"{'No':<5} | {'Success':<8} | {'Time(s)':<10} | {'Ratio':<8} | {'Items'}")
    print("-" * 60)

    success_count = 0
    total_start_time = time.time()
    
    for i in range(args.test_episode):
        obs, _ = test_env.reset()
        done = False
        raw_env = test_env.unwrapped
        total_items_to_pack = len(raw_env.box_creator.all_items)
        
        ep_start_time = time.perf_counter()
        
        while not done:
            batch = Batch(obs=[obs], info={})
            with torch.no_grad():
                result = policy(batch)
            
            action = result.act[0]
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

        ep_end_time = time.perf_counter()
        duration = ep_end_time - ep_start_time

        # T/F 판단
        packed_count = info['counter']
        is_success = (packed_count == total_items_to_pack)
        if is_success: success_count += 1
        
        print(f"{i+1:<5} | {str(is_success):<8} | {duration:<10.4f} | {info['ratio']:.4f} | {packed_count}/{total_items_to_pack}")

    total_duration = time.time() - total_start_time
    print("-" * 60)
    print(f"Final Success Rate: {success_count}/{args.test_episode} ({success_count/args.test_episode*100:.1f}%)")
    print(f"Total Test Time: {total_duration:.2f} seconds")
    
if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.ckp = r"C:\Users\USER\Desktop\SDO\policy_step_best6.pth"
    args.test_episode = 10 
    args.seed = int(time.time())
    
    test_tf(args)