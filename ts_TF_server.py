import os
import sys
import argparse
import torch
import gymnasium as gym
import numpy as np
import json
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from tianshou.data import Batch

from cvrp_utils import CVRPParser
from ts_train import build_net
from masked_ppo import MaskedPPOPolicy
from tools import registration_envs, set_seed, CategoricalMasked
import arguments

class PackingServer:
    def __init__(self, args, file_id):
        self.args = args
        self.file_id = str(file_id).zfill(2)
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        set_seed(args.seed, args.cuda, args.cuda_deterministic)
        
        print(f"[*] Loading model from {args.ckp}...")
        self.actor, self.critic = build_net(args, self.device)
        self.policy = self._init_policy()
        self.policy.eval()
        
        self.parser, self.env = self._load_resources()

    def _init_policy(self):
        optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.opt.lr)
        policy = MaskedPPOPolicy(
            actor=self.actor, critic=self.critic, optim=optim,
            dist_fn=CategoricalMasked, action_space=gym.spaces.Discrete(self.args.env.k_placement * 2)
        )
        
        if not os.path.exists(self.args.ckp):
            print(f"Error: Model file not found at {self.args.ckp}")
            sys.exit()
            
        policy.load_state_dict(torch.load(self.args.ckp, map_location=self.device))
        return policy

    def _load_resources(self):
        file_path = rf"C:\Users\USER\Desktop\SDO\GOPT_cvrp\3L_CVRP\3l_cvrp{self.file_id}.txt"
        if not os.path.exists(file_path):
            print(f"Error: Data file not found at {file_path}")
            sys.exit()
        
        parser = CVRPParser(file_path)
        veh_info = parser.vehicle_info
        container_size = (veh_info['length'], veh_info['width'], veh_info['height'])
        
        env = gym.make(
            self.args.env.id,
            container_size=container_size,
            enable_rotation=self.args.env.rot,
            data_type="cvrp", 
            item_set=None,
            reward_type=self.args.train.reward_type,
            action_scheme=self.args.env.scheme,
            k_placement=self.args.env.k_placement,
            is_render=False,
            cvrp_parser=parser 
        )
        print(f"[*] File {self.file_id} loaded successfully.")
        return parser, env

    def handle_request(self, data):
        route_ids = data['route']
        print(f"\n[Request] Processing Route: {route_ids}")
        
        # 1. 환경 초기화
        self.env.reset()
        
        # 2. 요청받은 경로를 강제로 환경(BoxCreator)에 주입
        box_creator = self.env.unwrapped.box_creator
        box_creator.current_route = route_ids
        box_creator.node_items = []
        box_creator.total_route_items = 0
        
        for node_id in reversed(route_ids):
            items_in_node = self.parser.items.get(node_id, [])
            if items_in_node:
                box_creator.node_items.append(list(items_in_node))
                box_creator.total_route_items += len(items_in_node)
                
        if not box_creator.node_items:
            box_creator.node_items = [[(0, 0, 0)]]
            box_creator.total_route_items = 0
            
        box_creator.current_node_idx = 0
        
        # 3. 주입된 데이터에 맞게 최신 관측값(obs) 갱신
        obs_dict = self.env.unwrapped.cur_observation
        terminated = False
        
        # 4. 에이전트 행동 반복 수행
        while not terminated:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_dict['obs']).float().unsqueeze(0).to(self.device)
                mask_tensor = torch.from_numpy(obs_dict['mask']).unsqueeze(0).to(self.device)
                input_batch = Batch(obs=obs_tensor, mask=mask_tensor)
                
                logits, _ = self.policy.actor(input_batch)
                logits[mask_tensor == 0] = -1e10
                action = logits.argmax(dim=1).cpu().item()

            obs_dict, reward, terminated, truncated, info = self.env.step(action)

        # 5. 최종 결과 추출 (env.py에서 info에 담아준 값 활용)
        is_success = info.get('is_success', False)
        packed = info.get('counter', 0)
        total = info.get('total_items', 0)
        
        status = "SUCCESS" if is_success else "FAILED"
        print(f"[Result] {status} | Route packed: {packed}/{total}")
        
        return {"result": is_success, "packed": packed, "total": total}

def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=int, required=True, help="파일 ID 지정")
    parser.add_argument('--port', type=int, default=9999)
    # 기존 인자와 충돌하지 않도록 모르는 인자는 분리
    cmd_args, unknown = parser.parse_known_args()

    registration_envs()
    args = arguments.get_args()
    args.ckp = r"C:\Users\USER\Desktop\SDO\learned_model\policy_step_best7.pth"
    
    server_logic = PackingServer(args, cmd_args.file)
    
    with socket(AF_INET, SOCK_STREAM) as s:
        s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', cmd_args.port))
        s.listen()
        print(f"[*] Server listening on port {cmd_args.port} with File {cmd_args.file}...")
        
        while True:
            conn, addr = s.accept()
            with conn:
                raw_data = conn.recv(4096).decode('utf-8')
                if not raw_data: continue
                request_json = json.loads(raw_data)
                response = server_logic.handle_request(request_json)
                conn.sendall(json.dumps(response).encode('utf-8'))

if __name__ == '__main__':
    run_server()