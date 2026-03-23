import numpy as np
import random
import math

class CVRPParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.vehicle_info = {} 
        self.nodes = {}        
        self.items = {}        
        self.num_customers = 0
        
        self._parse()

    def _parse(self):
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            
            if "number of customers" in line:
                self.num_customers = int(line.split()[0])
            elif "number of vehicles" in line:
                self.vehicle_info['count'] = int(line.split()[0])
            elif "Capacity - height - width - length" in line:
                idx += 1
                vals = list(map(int, lines[idx].split()))
                self.vehicle_info['capacity'] = vals[0] 
                self.vehicle_info['height'] = vals[1]
                self.vehicle_info['width'] = vals[2]
                self.vehicle_info['length'] = vals[3]

            elif "Node - x - y - demand" in line:
                for _ in range(self.num_customers + 1): 
                    idx += 1
                    node_vals = list(map(float, lines[idx].split()))
                    node_id = int(node_vals[0])
                    self.nodes[node_id] = {
                        'x': node_vals[1], 
                        'y': node_vals[2], 
                        'demand': node_vals[3] 
                    }
                    self.items[node_id] = []

            elif "Node - number of items - h - w - l" in line:
                idx += 1
                while idx < len(lines):
                    item_line = lines[idx]
                    if "Instance" in item_line: 
                        break
                    
                    parts = list(map(str, item_line.split()))
                    node_id = int(parts[0])
                    
                    part_idx = 2
                    while part_idx < len(parts):
                        h = int(parts[part_idx])
                        w = int(parts[part_idx+1])
                        l = int(parts[part_idx+2])
                        self.items[node_id].append((l, w, h))
                        part_idx += 4
                    idx += 1
                continue 
            
            idx += 1

def generate_random_routes(parser: CVRPParser, num_generated_sets: int):
    """
    아이템 개수 기반 경로 생성 (모든 노드 방문 보장)
    """
    max_vehicles = parser.vehicle_info['count']
    customer_ids = list(range(1, parser.num_customers + 1))
    
    # 1. 총 아이템 개수 계산 및 차량별 한도 설정 (올림)
    total_items = sum(len(parser.items.get(node_id, [])) for node_id in customer_ids)
    item_limit = math.ceil(total_items / max_vehicles)
    
    valid_route_sets = []

    while len(valid_route_sets) < num_generated_sets:
        random.shuffle(customer_ids)
        
        current_set = []
        current_route = []
        current_item_count = 0
        is_success = True
        
        for customer in customer_ids:
            node_items = parser.items.get(customer, [])
            node_item_count = len(node_items)
            
            # 2. 한도 초과 시 다음 차량으로 넘김 (노드 분할 금지)
            if current_item_count + node_item_count <= item_limit:
                current_route.append(customer)
                current_item_count += node_item_count
            else:
                current_set.append(current_route)
                
                # 3. 차량 대수를 초과했는데 노드가 남은 경우 실패 처리
                if len(current_set) >= max_vehicles:
                    is_success = False
                    break
                
                current_route = [customer]
                current_item_count = node_item_count
        
        if is_success:
            if current_route:
                current_set.append(current_route)
            
            # 차량 대수 유지를 위해 빈 리스트 추가
            while len(current_set) < max_vehicles:
                current_set.append([])
                
            valid_route_sets.append(current_set)
            
    return valid_route_sets

def get_items_for_route_reversed(parser: CVRPParser, route: list):
    loading_sequence = []
    for node_id in reversed(route):
        node_items = parser.items.get(node_id, [])
        loading_sequence.extend(node_items)
    return loading_sequence