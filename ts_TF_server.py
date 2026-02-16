import os
import sys
import argparse
import torch
import gymnasium as gym
import numpy as np
import json
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from tianshou.data import Batch

# 기존 프로젝트 모듈 임포트
from cvrp_utils import CVRPParser, get_items_for_route_reversed
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
        
        # 서버 시작 시 파일을 미리 로드
        self.parser, self.env = self._load_resources()

    def _init_policy(self):
        optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.opt.lr)
        policy = MaskedPPOPolicy(
            actor=self.actor, critic=self.critic, optim=optim,
            dist_fn=CategoricalMasked, action_space=gym.spaces.Discrete(self.args.env.k_placement * 2)
        )
        policy.load_state_dict(torch.load(self.args.ckp, map_location=self.device))
        return policy

    def _load_resources(self):
        file_path = f"C:/Users/USER/Desktop/SDO/GOPT_cvrp/3L_CVRP/3l_cvrp{self.file_id}.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        parser = CVRPParser(file_path)
        veh_info = parser.vehicle_info
        container_size = (veh_info['length'], veh_info['width'], veh_info['height'])
        
        env = gym.make(
            self.args.env.id,
            container_size=container_size,
            enable_rotation=self.args.env.rot,
            data_type="random",
            item_set=[(1,1,1)],
            reward_type=self.args.train.reward_type,
            action_scheme=self.args.env.scheme,
            k_placement=self.args.env.k_placement,
            is_render=False
        )
        print(f"[*] File {self.file_id} loaded successfully.")
        return parser, env

    def handle_request(self, data):
        route_ids = data['route']
        print(f"\n[Request] Processing Route: {route_ids}")
        
        # 1. 고객(Node)별로 아이템을 가져와서 어떤 고객의 것인지 추적할 수 있게 구성
        # get_items_for_route_reversed 대신 내부 로직을 직접 사용하여 노드 정보를 남깁니다.
        obs, _ = self.env.reset()
        
        step_count = 1
        for node_id in reversed(route_ids):
            node_items = self.parser.items.get(node_id, []) # 해당 고객의 아이템 리스트 
            
            for i, item in enumerate(node_items):
                # item은 (l, w, h) 튜플임 
                print(f"  > Step {step_count} | Customer {node_id} - Item {i+1} {item}...", end=" ")
                
                self.env.unwrapped.box_creator.box_list = [item]
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs['obs']).float().unsqueeze(0).to(self.device)
                    mask_tensor = torch.from_numpy(obs['mask']).unsqueeze(0).to(self.device)
                    input_batch = Batch(obs=obs_tensor, mask=mask_tensor)
                    
                    logits, _ = self.policy.actor(input_batch)
                    logits[mask_tensor == 0] = -1e10
                    action = logits.argmax(dim=1).cpu().item()

                obs, _, terminated, _, _ = self.env.step(action)
                
                if terminated:
                    print(f"FAILED (Action: {action})")
                    print(f"[Result] Route failed at Customer {node_id}, Item {i+1}")
                    return {"result": False, "failed_node": node_id, "item_idx": i+1}
                
                print(f"OK (Action: {action})")
                step_count += 1

        print("[Result] Route successfully packed!")
        return {"result": True}

def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=int, required=True, help="파일 ID 지정")
    parser.add_argument('--port', type=int, default=9999)
    cmd_args = parser.parse_args()

    registration_envs()
    args = arguments.get_args()
    args.ckp = r"C:\Users\USER\Desktop\SDO\policy_step_best6.pth"
    
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