import numpy as np
import copy
import torch
# binCreator.py 상단에 import 추가
import random
# cvrp_utils.py가 같은 폴더에 있어야 합니다.
from cvrp_utils import CVRPParser, generate_random_routes, get_items_for_route_reversed


class BoxCreator(object):
    def __init__(self):
        self.box_list = []  # generated box list

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        """
        :param length:
        :return: list
        """
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)


class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                default_box_set.append((2 + i, 2 + j, 2 + k))

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set
        # print(self.box_set)

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])


# load data
class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):  # data url
        super().__init__()  
        self.data_name = data_name
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))  
        print("load data set successfully, data name: ", self.data_name)

    def reset(self, index=None):
        self.box_list.clear()
        box_trajs = torch.load(self.data_name)
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = box_trajs[self.index]
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1

class CVRPBoxCreator(BoxCreator):
    def __init__(self, cvrp_parser: CVRPParser):
        super().__init__()
        self.parser = cvrp_parser
        self.box_list = []
        self.box_index = 0
        
    def reset(self):
        self.box_list.clear()
        self.box_index = 0
        
        # 1. 학습을 위해 매번 새로운 랜덤 경로 생성
        # generate_random_routes는 [Set_1, Set_2...]를 반환하므로 하나만 생성해서 씁니다.
        # 여기서는 차량 1대의 경로 하나만 샘플링하여 학습에 사용합니다 (단일 차량 패킹 문제로 환원)
        route_sets = generate_random_routes(self.parser, 1) # 1세트 생성
        vehicle_routes = route_sets[0] # [Route_Veh1, Route_Veh2, ...]
        
        # 빈 경로가 아닌 것 중 하나를 랜덤 선택 (다양성 확보)
        valid_routes = [r for r in vehicle_routes if len(r) > 0]
        if not valid_routes:
            # 만약 모든 차량 경로가 비었다면(고객 0명) Depot만 있는 경우 등
            target_route = []
        else:
            target_route = random.choice(valid_routes)
            
        # 2. 경로의 역순(LIFO)으로 아이템 리스트 가져오기
        # items: List of (l, w, h) -> GOPT는 (x, y, z)로 사용
        self.all_items = get_items_for_route_reversed(self.parser, target_route)
        
        # 3. 초기 박스 로딩 (빈 리스트일 경우 처리)
        if not self.all_items:
             self.all_items = [(0,0,0)] # 더미

    def generate_box_size(self, **kwargs):
        """환경에서 drop_box() 호출 후 다음 박스를 준비할 때 사용"""
        if self.box_index < len(self.all_items):
            self.box_list.append(self.all_items[self.box_index])
            self.box_index += 1
        else:
            # 아이템 소진 시 (0,0,0)을 주어 환경이 종료 조건을 감지하게 함
            # (env.py의 step 함수 로직에 따라 다를 수 있으나 보통 크기가 0이면 무시되거나 종료됨)
            self.box_list.append((1000, 1000, 1000)) # 혹은 매우 큰 값을 주어 실패하게 하여 종료 유도
            # *참고*: GOPT env.py는 보통 place_box 실패 시 종료됩니다.