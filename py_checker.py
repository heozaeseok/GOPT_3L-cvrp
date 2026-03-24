import os
import sys
import torch
import gymnasium as gym
from tianshou.data import Batch

from ts_train import build_net
import arguments
from tools import set_seed, CategoricalMasked
from masked_ppo import MaskedPPOPolicy
from cvrp_utils import CVRPParser
from binCreator import EvalBoxCreator



#전역 초기화 (C++에서 import 시 1회만 실행)
args = arguments.get_args()
args.train.algo = args.train.algo.upper()

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
set_seed(args.seed, args.cuda, args.cuda_deterministic)

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cvrp_file = os.path.join(BASE_DIR, "3L_CVRP", "3l_cvrp02.txt")
args.ckp = os.path.join(BASE_DIR, "learned_model", "ver2", "policy_step_best_0310.pth")

parser = CVRPParser(cvrp_file)
veh_info = parser.vehicle_info
args.env.container_size = (veh_info['length'], veh_info['width'], veh_info['height'])

# 환경 생성
test_env = gym.make(
    args.env.id, 
    container_size=args.env.container_size,
    enable_rotation=args.env.rot,
    data_type="cvrp",
    item_set=None,
    reward_type=args.train.reward_type,
    action_scheme=args.env.scheme,
    k_placement=args.env.k_placement,
    is_render=False, 
    cvrp_parser=parser
)

# 생성기를 EvalBoxCreator로 교체
test_env.unwrapped.box_creator = EvalBoxCreator(parser)

# 모델 구성 및 로드
actor, critic = build_net(args, device)
optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.opt.lr)

policy = MaskedPPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=CategoricalMasked,
    discount_factor=args.train.gamma,
    eps_clip=args.train.clip_param,
    action_space=test_env.action_space,
)

policy.load_state_dict(torch.load(args.ckp, map_location=device))
policy.eval()


#C++ 호출용 check 함수
def check(route):
    # C++에서 전달받은 route 주입
    test_env.unwrapped.box_creator.set_route(route)
    
    # 주입된 route로 환경 리셋
    obs, info = test_env.reset()
    done = False
    
    # 에피소드 실행
    while not done:
        batch = Batch(obs=[obs], info=[info])
        
        with torch.no_grad():
            result = policy(batch)
            
        action = result.act[0].item()
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
    return info.get('is_success', False)
