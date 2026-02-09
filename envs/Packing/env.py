from typing import Optional

from .container import Container
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .cutCreator import CuttingBoxCreator
# from .mdCreator import MDlayerBoxCreator
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator

from render import VTKRender
# env.py

# 상단 import에 추가
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator, CVRPBoxCreator
# cvrp_utils 필요시 import (parser 객체를 인자로 넘길 것이므로 필수는 아님)


class PackingEnv(gym.Env):
    def __init__(
        self,
        container_size=(10, 10, 10),
        item_set=None, 
        data_name=None, 
        load_test_data=False,
        enable_rotation=False,
        data_type="random",
        reward_type=None,
        action_scheme="heightmap",
        k_placement=100,
        is_render=False,
        is_hold_on=False,
        cvrp_parser=None,
        **kwags
    ) -> None:
        self.bin_size = container_size
        self.area = int(self.bin_size[0] * self.bin_size[1])
        # packing state
        self.container = Container(*self.bin_size, rotation=enable_rotation)
        self.can_rotate = enable_rotation   
        self.reward_type = reward_type
        self.action_scheme = action_scheme
        self.k_placement = k_placement
        if action_scheme == "EMS":
            self.candidates = np.zeros((self.k_placement, 6), dtype=np.int32)  # (x1, y1, z1, x2, y2, H)
        else:
            self.candidates = np.zeros((self.k_placement, 3), dtype=np.int32)  # (x, y, z)

        # Generator for train/test data
        if not load_test_data:
            # [수정] data_type이 'cvrp'가 아닐 때만 item_set 확인
            if data_type != "cvrp":
                assert item_set is not None

            if data_type == "random":
                print(f"using items generated randomly")
                self.box_creator = RandomBoxCreator(item_set)  
            if data_type == "cut":
                print(f"using items generated through cutting method")
                low = list(item_set[0])
                up = list(item_set[-1])
                low.extend(up)
                self.box_creator = CuttingBoxCreator(container_size, low, self.can_rotate)
            # [추가] CVRP 데이터 타입 처리
            if data_type == "cvrp":
                print(f"using items generated from CVRP routes")
                assert cvrp_parser is not None, "CVRP Parser must be provided for cvrp data_type"
                self.box_creator = CVRPBoxCreator(cvrp_parser)

            assert isinstance(self.box_creator, BoxCreator)
        if load_test_data:
            print(f"use box dataset: {data_name}")
            self.box_creator = LoadBoxCreator(data_name)

        self.test = load_test_data

        # for rendering
        if is_render:
            self.renderer = VTKRender(container_size, auto_render=not is_hold_on)
        self.render_box = None
        
        self._set_space()

    def _set_space(self) -> None:
        obs_len = self.area + 3 
        obs_len += self.k_placement * 6
        # [수정] 위치(k_placement) x 회전(2) = 2 * k_placement
        self.action_space = spaces.Discrete(self.k_placement * 2) 
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=max(self.bin_size), shape=(obs_len, )),
                "mask": spaces.Box(low=0, high=1, shape=(self.k_placement * 2,), dtype=np.int8)
            }
        )

    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (
                self.container.dimension[0] * self.container.dimension[1] * self.container.dimension[2])

    # box mask (W x L x 3)
    def get_box_plain(self):
        coming_box = self.next_box
        x_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[0]
        y_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[1]
        z_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[2]
        return x_plain, y_plain, z_plain

    @property
    def cur_observation(self):
        """
            get current observation and action mask
        """
        hmap = self.container.heightmap
        size = list(self.next_box)
        placements, mask = self.get_possible_position(size)
        self.candidates = np.zeros_like(self.candidates)
        if len(placements) != 0:
            # print("candidates:")
            # for c in placements:
            #     print(c)
            self.candidates[0:len(placements)] = placements

        size.extend([size[1], size[0], size[2]])
        obs = np.concatenate((hmap.reshape(-1), np.array(size).reshape(-1), self.candidates.reshape(-1)))
        mask = mask.reshape(-1)
        return {
            "obs": obs, 
            "mask": mask
        }

    @property
    def next_box(self) -> list:
        return self.box_creator.preview(1)[0]
    
    def get_possible_position(self, next_box):
        if self.action_scheme == "heightmap":
            candidates = self.container.candidate_from_heightmap(next_box, self.k_placement)
            mask = np.ones((2, len(candidates)), dtype=np.int8) 
        elif self.action_scheme == "EP":
            candidates, mask = self.container.candidate_from_EP(next_box, self.k_placement)
        elif self.action_scheme == "EMS":
            candidates, mask = self.container.candidate_from_EMS(next_box, self.k_placement)
        elif self.action_scheme == "FC":
            candidates, mask = self.container.candidate_from_FC(next_box)
        else:
            raise NotImplementedError("action scheme not implemented")

        hmap = self.container.heightmap
        
        for i, ems in enumerate(candidates):
            x_start, y_start, z_base = int(ems[0]), int(ems[1]), int(ems[2])
            # [수정] 입구가 X_max이므로 기존 x_start == 0 예외 처리는 삭제

            for rot in range(2):
                if mask[rot, i] == 0: continue 

                # 회전 여부에 따른 현재 박스의 X, Y 크기 결정
                curr_size_x = next_box[0] if rot == 0 else next_box[1]
                curr_size_y = next_box[1] if rot == 0 else next_box[0]
                
                # [수정] 박스가 입구(X_max)에 딱 붙어 적재되는 경우 경로 체크 불필요
                if x_start + curr_size_x >= hmap.shape[0]:
                    continue

                y_end = min(y_start + curr_size_y, hmap.shape[1])
                
                # [수정] 현재 배치 위치(x_start + curr_size_x)부터 입구(hmap 끝)까지의 경로 확인
                path_area = hmap[x_start + curr_size_x:, y_start:y_end]
                
                if np.any(path_area > z_base):
                    mask[rot, i] = 0

        # 모든 위치가 불가능할 경우에 대한 폴백 로직
        if np.all(mask == 0):
            mask[0, 0] = 1

        return candidates, mask

    def idx2pos(self, idx):
        # k_placement 단위로 rot(회전) 여부를 결정
        if idx >= self.k_placement:
            idx = idx - self.k_placement
            rot = 1
        else:
            rot = 0

        pos = self.candidates[idx][:3]

        if rot == 1:
            dim = [self.next_box[1], self.next_box[0], self.next_box[2]]
        else:
            dim = list(self.next_box)
        self.render_box = [dim, pos]

        return pos, rot, dim
    
    def step(self, action):
        """

        :param action: action index
        :return: cur_observation
                 reward
                 done, Whether to end boxing (i.e., the current box cannot fit in the bin)
                 info
        """
        # print(self.next_box)
        pos, rot, size = self.idx2pos(action)
 
        succeeded = self.container.place_box(self.next_box, pos, rot)
        
        if not succeeded:
            if self.reward_type == "terminal":  # Terminal reward
                reward = self.container.get_volume_ratio()
            else:  # Step-wise/Immediate reward
                reward = 0.0
            done = True
            
            self.render_box = [[0, 0, 0], [0, 0, 0]]
            info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
            return self.cur_observation, reward, done, False, info

        box_ratio = self.get_box_ratio()

        self.box_creator.drop_box()  # remove current box from the list
        self.box_creator.generate_box_size()  # add a new box to the list

        if self.reward_type == "terminal":
            reward = 0.01
        else:
            reward = box_ratio
        done = False
        info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}

        return self.cur_observation, reward, done, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.box_creator.reset()
        self.container = Container(*self.bin_size)
        self.box_creator.generate_box_size()
        self.candidates = np.zeros_like(self.candidates)
        return self.cur_observation, {}
    
    def seed(self, s=None):
        np.random.seed(s)

    def render(self):
        self.renderer.add_item(self.render_box[0], self.render_box[1])
        # self.renderer.save_img()