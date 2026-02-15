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
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        set_seed(args.seed, args.cuda, args.cuda_deterministic)
        
        print(f"[*] Loading model from {args.ckp}...")
        self.actor, self.critic = build_net(args, self.device)
        self.policy = self._init_policy()
        self.policy.eval()
        
        self.parsers = {}
        self.envs = {}

    def _init_policy(self):
        optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.opt.lr)
        policy = MaskedPPOPolicy(
            actor=self.actor, critic=self.critic, optim=optim,
            dist_fn=CategoricalMasked, action_space=gym.spaces.Discrete(self.args.env.k_placement * 2)
        )
        policy.load_state_dict(torch.load(self.args.ckp, map_location=self.device))
        return policy

    def get_resources(self, file_id):
        if file_id not in self.parsers:
            file_path = f"C:/Users/USER/Desktop/SDO/GOPT_cvrp/3L_CVRP/3l_cvrp{file_id}.txt"
            if not os.path.exists(file_path):
                return None, None
            
            parser = CVRPParser(file_path)
            self.parsers[file_id] = parser
            
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
            self.envs[file_id] = env
            
        return self.parsers[file_id], self.envs[file_id]

    def handle_request(self, data):
        file_id = str(data['file']).zfill(2)
        route_ids = data['route']
        
        parser, env = self.get_resources(file_id)
        if parser is None:
            return {"result": False, "error": "File not found"}

        loading_items = get_items_for_route_reversed(parser, route_ids)
        obs, _ = env.reset()
        
        for item in loading_items:
            env.unwrapped.box_creator.box_list = [item]
            
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs['obs']).float().unsqueeze(0).to(self.device)
                mask_tensor = torch.from_numpy(obs['mask']).unsqueeze(0).to(self.device)
                input_batch = Batch(obs=obs_tensor, mask=mask_tensor)
                
                logits, _ = self.policy.actor(input_batch)
                logits[mask_tensor == 0] = -1e10
                action = logits.argmax(dim=1).cpu().item()

            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                return {"result": False}

        return {"result": True}

def run_server(host='127.0.0.1', port=9999):
    registration_envs()
    args = arguments.get_args()
    args.ckp = r"C:\Users\USER\Desktop\SDO\policy_step_best6.pth"
    
    server_logic = PackingServer(args)
    
    with socket(AF_INET, SOCK_STREAM) as s:
        s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        print(f"[*] Server started at {host}:{port}. Waiting for C client...")
        
        while True:
            conn, addr = s.accept()
            with conn:
                raw_data = conn.recv(4096).decode('utf-8')
                if not raw_data: continue
                
                print(f"[*] Received: {raw_data}")
                request_json = json.loads(raw_data)
                response = server_logic.handle_request(request_json)
                
                conn.sendall(json.dumps(response).encode('utf-8'))

if __name__ == '__main__':
    run_server()