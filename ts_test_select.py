import os
import sys
import ast
import torch
import gymnasium as gym

# 1. 경로 설정: binCreator.py가 있는 폴더를 검색 경로에 추가
curr_path = os.path.dirname(os.path.abspath(__file__))
# 알려주신 경로 반영: envs/Packing 폴더 추가
packing_path = os.path.join(curr_path, "envs", "Packing")
if packing_path not in sys.path:
    sys.path.append(packing_path)

# 프로젝트 루트도 추가
if curr_path not in sys.path:
    sys.path.append(curr_path)

# 이제 모듈을 불러옵니다.
from binCreator import EvalBoxCreator
from cvrp_utils import CVRPParser

from ts_train import build_net
import arguments
from tools import *
from mycollector import PackCollector
from masked_ppo import MaskedPPOPolicy
from tianshou.utils.net.common import ActorCritic

def test(args, target_route):
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")
        
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # 데이터 로드
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        exit()
    parser = CVRPParser(args.data_path)
    
    # 환경 설정
    veh_info = parser.vehicle_info
    args.env.container_size = (veh_info['length'], veh_info['width'], veh_info['height'])
    
    test_env = gym.make(
        args.env.id, 
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type="cvrp",            
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=args.render,
        cvrp_parsers=[parser]        
    )

    # EvalBoxCreator에 경로 주입 및 환경 연결
    custom_creator = EvalBoxCreator(parser)
    custom_creator.set_route(target_route)
    test_env.unwrapped.box_creator = custom_creator 
    
    test_env.reset()

    # 모델 설정
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    
    policy = MaskedPPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=CategoricalMasked,
        discount_factor=args.train.gamma, eps_clip=args.train.clip_param,
        advantage_normalization=False, vf_coef=args.loss.value,
        ent_coef=args.loss.entropy, gae_lambda=args.train.gae_lambda,
        action_space=test_env.action_space,
    )
    
    policy.eval()
    print(f"Loading model from: {args.ckp}")
    policy.load_state_dict(torch.load(args.ckp, map_location=device))

    # 테스트 및 시각화 실행
    test_collector = PackCollector(policy, test_env)
    print(f"\n[시각화] 입력 경로 {target_route} 적재 시작 (LIFO 순서)...")
    
    result = test_collector.collect(n_episode=1, render=args.render)
    
    print('----------------------------------------------')
    print(f"결과 - 공간 효율성: {result['ratio']:.4f} | 적재 개수: {result['num']}")

if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()

    # 하드코딩 경로 설정
    args.data_path = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\3L_CVRP\3l_cvrp01.txt"
    args.ckp = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\learned_model\ver2\policy_step_best_0402.pth"
    args.render = True
    args.test_episode = 1

    print("\n입력 예시: [6, 13, 4, 12]")
    user_input = input("Route를 입력하세요: ").strip()
    
    # 예외 처리 없이 바로 리스트로 변환
    target_route = ast.literal_eval(user_input)
    
    import time
    args.seed = int(time.time()) 

    test(args, target_route)