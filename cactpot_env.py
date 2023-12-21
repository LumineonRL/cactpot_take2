from typing import Tuple, List, Dict, Optional
import gymnasium as gym
from gym import spaces
from gym import utils
import numpy as np

class CactPot(gym.Env):
    metadata = {'render.modes': ['human']}
    PAYOUT = {
        6: 10000,
        7: 36,
        8: 720,
        9: 360,
        10: 80,
        11: 252,
        12: 108,
        13: 72,
        14: 54,
        15: 180,
        16: 72,
        17: 180,
        18: 119,
        19: 36,
        20: 306,
        21: 1080,
        22: 144,
        23: 1800,
        24: 3600
    }
        
    def __init__(self):
        super(CactPot, self).__init__()
        self.max_selections = 3
        self.grid_size = 3
        self.num_values = 9
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, self.num_values), dtype=np.int32),
            spaces.Discrete(2)
        ))
        self.true_grid = None
        self.one_hot_grid = None
        self.revealed_count = 0
        self.selected_line = None
        self.reward = 0
        self.max_penalty = -max(self.PAYOUT.values())
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        self.true_grid = self._generate_unique_numbers()
        self.one_hot_grid = np.zeros((self.grid_size, self.grid_size, self.num_values), dtype=np.int32)
        self.revealed_count = 0
        self.selected_line = None
        self.reward = 0
        
        return (self.one_hot_grid, 0 if self.revealed_count < self.max_selections else 1)
    
    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], int, bool, dict]:
        if self.revealed_count < self.max_selections:
            if 1 <= action + 1 <= 9 and not self._is_revealed(action):
                self._reveal_number(action)
            else:
                self.reward += self.max_penalty
        else:
            if 0 <= action < 8 and not self.selected_line:
                self._process_line_selection(action)
            else:
                self.reward += self.max_penalty

        done = self.selected_line is not None
        if done:
            self._reveal_all()

        return (self.one_hot_grid, 0 if self.revealed_count < self.max_selections else 1), self.reward, done, {}

    def _is_revealed(self, action: int) -> bool:
        row, col = self._get_grid_position(action)
        return self.one_hot_grid[row, col].sum() > 0

    def _process_line_selection(self, action: int) -> None:
        line = self._get_line(action)
        if line is not None:
            self._calculate_line_reward(line)
            self.selected_line = True

    def _get_line(self, action: int) -> Optional[np.ndarray]:
        if action < 3:
            return self.true_grid[action, :]
        elif action < 6:
            return self.true_grid[:, action - 3]
        elif action == 6:
            return self.true_grid.diagonal()
        elif action == 7:
            return self._get_secondary_diagonal()
        return None

    def _calculate_line_reward(self, line: np.ndarray) -> None:
        line_sum = np.sum(line.astype(int))
        self.reward = self.PAYOUT.get(line_sum, 0)

    def render(self, mode: str = 'human', show_true_grid: bool = False) -> None:
        if mode == 'human':
            if show_true_grid:
                print("True Grid:")
                print('\n'.join([' '.join(map(str, row)) for row in self.true_grid]))
            else:
                human_readable_grid = self._get_human_readable_grid()
                print('\n'.join([' '.join(row) for row in human_readable_grid]))
        else:
            print(self.one_hot_grid)

    def _get_human_readable_grid(self) -> np.ndarray:
        human_readable_grid = np.full((self.grid_size, self.grid_size), '?', dtype=np.str_)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.one_hot_grid[row, col].sum() == 1:
                    number = np.argmax(self.one_hot_grid[row, col]) + 1
                    human_readable_grid[row, col] = str(number)
        return human_readable_grid

    def _generate_unique_numbers(self) -> np.ndarray:
        numbers = self.np_random.permutation(9) + 1
        return numbers.reshape(self.grid_size, self.grid_size)

    def _reveal_number(self, action: int) -> None:
        row, col = self._get_grid_position(action)
        number = self.true_grid[row, col] - 1
        self.one_hot_grid[row, col, number] = 1
        self.revealed_count += 1

    def _select_line(self, action: int) -> Optional[np.ndarray]:
        if action < 3:
            line = self.true_grid[action, :]
        elif action < 6:
            line = self.true_grid[:, action - 3]
        elif action == 6:
            line = self.true_grid.diagonal()
        elif action == 7:
            line = self._get_secondary_diagonal()
        else:
            return  # Invalid action, do nothing

        line_sum = np.sum(line.astype(int))
        self.reward = self.PAYOUT.get(line_sum, 0)  # Get reward based on line sum
        self.selected_line = True

    def _get_secondary_diagonal(self) -> np.ndarray:
        return np.array([self.true_grid[i, self.grid_size - 1 - i] for i in range(self.grid_size)])

    def  _get_grid_position(self, action: int) -> Tuple[int, int]:
        return action // self.grid_size, action % self.grid_size

    def _reveal_all(self) -> None:
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                number = self.true_grid[row, col] - 1
                self.one_hot_grid[row, col, number] = 1

    def _get_observation(self) -> Dict[str, np.ndarray]:
        return {
            'grid': self.one_hot_grid,
            'phase': 0 if self.revealed_count < self.max_selections else 1
        }