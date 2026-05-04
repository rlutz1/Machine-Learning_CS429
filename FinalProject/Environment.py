"""
Simulates agent movement on an abstracted 2-D map for RL training

Grid conventions:
  - 2-D numpy array where rows = y-axis, cols = x-axis
  - Pixel value >= 128  →  white (accessible node)
  - Pixel value <  128  →  black (obstacle node)

State representation: (x, y) tuple where x = column index, y = row index

Actions:
  0 = LEFT x - 1
  1 = RIGHT x + 1
  2 = UP y - 1
  3 = DOWN y + 1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Actions
LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
ACTION_NAMES = {LEFT: "LEFT", RIGHT: "RIGHT", UP: "UP", DOWN: "DOWN"}
N_ACTIONS = 4

# Changes needed to move in that direction
DELTA = {
    LEFT: (-1, 0),
    RIGHT: ( 1, 0),
    UP: ( 0, -1),
    DOWN: ( 0, 1),
}


class Environment:
    """
    grid : 2-D numpy array
    target : (int, int) – target position as (x, y)
    """

    def __init__(self, grid: np.ndarray, target: tuple):
        self.grid = grid
        self.n_rows, self.n_cols = grid.shape
        self.target = target

        self.S3_rewards = {}
        self._init_S3_rewards() # specific to S3 for ease of grabbing.

        # Validate target is reachable
        tx, ty = target
        assert self.in_bounds(tx, ty), "Target is outside the map boundary."
        assert not self.is_obstacle(tx, ty), "Target sits on an obstacle."

    # Helpers

    def in_bounds(self, x: int, y: int) -> bool:
        """Return True if (x, y) is inside the map."""
        return 0 <= x < self.n_cols and 0 <= y < self.n_rows

    def is_obstacle(self, x: int, y: int) -> bool:
        """
        Return True if the cell is an obstacle region.
        Works for both 0-255 grayscale and binary
        """
        value = self.grid[y, x]
        # Normalise: if grid values are 0/1 treat 0 as obstacle, else use 128 threshold
        if self.grid.max() <= 1:
            return value == 0
        return value < 128

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not self.is_obstacle(x, y)

    # Reward the man
    def reward_S1(self, x: int, y: int, hit_boundary: bool) -> float:
        """
        Strategy 1
          +100 : reached the target
          -100 : hit an obstacle or went out of bounds
             0  : any other valid move
        """
        if (x, y) == self.target:
            return 100.0
        if hit_boundary:
            return -100.0
        return 0.0

    def reward_S2(self, x: int, y: int, hit_boundary: bool) -> float:
        """
        Strategy 2
          +100  : reached the target
          -100  : hit an obstacle or went out of bounds
            -1  : every normal step to encourage shorter path
        """
        if (x, y) == self.target:
            return 100.0
        if hit_boundary:
            return -100.0
        return -1.0
    

    def reward_S3(self, x: int, y: int, hit_boundary: bool) -> float:
        """
        Strategy 3
          +100                 : reached the target
          -100                 : hit an obstacle or went out of bounds
            -1                 : every normal step to encourage shorter path
          {0, 10, 20, ..., 90} : rippling larger and larger rewards the closer to target  
        """
        # if (x, y) == self.target:
        #     return 100.0
        if hit_boundary:
            return -100.0
        return self.S3_rewards[(x, y)]

    def compute_reward(self, x: int, y: int, hit_boundary: bool,
                        strategy: str) -> float:
        if strategy == "S1":
            return self.reward_S1(x, y, hit_boundary)
        elif strategy == "S2":
            return self.reward_S2(x, y, hit_boundary)
        elif strategy == "S3":
            return self.reward_S3(x, y, hit_boundary)
        else:
            raise ValueError(f"Unknown reward strategy")

    # Interface
    def step(self, state: tuple, action: int,
             strategy: str = "S1") -> tuple:
        """
        Execute one action from the given state.

        state : (x, y) current position
        action : integer in {0, 1, 2, 3} (LEFT / RIGHT / UP / DOWN)
        strategy : S1 or S2 can potentially add more?

        Returns
        next_state : (x, y)  – position after the action (unchanged if the move was illegal)
        reward: float – immediate reward
        done: bool – True if the episode has ended (target reached)
        crash: bool - True if obstacle hit/boundary hit
        """
        x, y = state
        dx, dy = DELTA[action]
        nx, ny = x + dx, y + dy

        # Bounds checking
        if not self.in_bounds(nx, ny):
            reward = self.compute_reward(x, y, hit_boundary=True, strategy=strategy)
            # Agent stays in current cell and episode ends
            return state, reward, False, True 

        # Obstacle
        if self.is_obstacle(nx, ny):
            reward = self.compute_reward(x, y, hit_boundary=True, strategy=strategy)
            # Treat obstacle collision same as boundary violation
            return state, reward, False, True 

        # Target reached
        next_state = (nx, ny)
        if next_state == self.target:
            reward = self.compute_reward(nx, ny, hit_boundary=False, strategy=strategy)
            return next_state, reward, True, False  # End here we win (yay)

        # Normal steppin
        reward = self.compute_reward(nx, ny, hit_boundary=False, strategy=strategy)
        return next_state, reward, False, False

    def get_free_cells(self) -> list:
        """Return a list of all accessible (x, y) positions on the map."""
        cells = []
        for y in range(self.n_rows):
            for x in range(self.n_cols):
                if self.is_free(x, y):
                    cells.append((x, y))
        return cells

    # Plot for some visuals

    def plot(self, path: list = None, title: str = "Map Abstraction"):
        """
        Display the map abstraction with the target position marked.
        path : optional list of (x, y) states representing a trajectory
        """
        fig, ax = plt.subplots(figsize=(7, 7))

        # Draw the grid (flip vertically so row 0 appears at the top)
        display = self.grid.copy().astype(float)
        if display.max() <= 1: # binary grid
            display = display * 255.0
        ax.imshow(display, cmap="gray", origin="upper", extent=[-0.5, self.n_cols - 0.5, self.n_rows - 0.5, -0.5])

        # Target marked with a green star
        tx, ty = self.target
        ax.plot(tx, ty, marker="*", color="lime", markersize=14, zorder=5, label="Target")

        # Optionally draw a trajectory
        if path is not None and len(path) > 1:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="red", linewidth=1.5,
                    marker="o", markersize=3, zorder=4, label="Path")
            # Mark start
            ax.plot(xs[0], ys[0], marker="s", color="cyan",
                    markersize=10, zorder=6, label="Start")

        ax.set_xlim(-0.5, self.n_cols - 0.5)
        ax.set_ylim(self.n_rows - 0.5, -0.5) # row 0 at top
        ax.set_xlabel("x (column)")
        ax.set_ylabel("y (row)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        plt.tight_layout()
        plt.show()


    # idea is to ripple out the rewards from the target
    # in order to better bait the agent,
    # but with care to avoid trapping it into an obstacle.
    def _init_S3_rewards(self):
        # setup all free cells as -1 to give some slight feedback
        free_cells = self.get_free_cells()
        for cell in free_cells:
            self.S3_rewards[cell] = -1 

        # setup the rippling effect
        self._ripple(self.target, 1000)

    # recursive algorithm to ripple out rewards
    def _ripple(self, cell: tuple, reward: float):
        # # base case: if not a free cell, stop
        # if not self.is_free(cell[0], cell[1]): 
        #     return

        # base case: if we've gone below a 0 reward, stop
        if reward < 0: return

        # general case: set this cell's reward to this reward
        self.S3_rewards[cell] = reward
        # update the reward to a lesser reward
        reward -= 100

        # then, recurse on all directions if the reward is still -1 (the default)
        
        dx, dy = DELTA[0]
        left_x, left_y = cell[0] + dx, cell[1] + dy

        dx, dy = DELTA[1]
        right_x, right_y = cell[0] + dx, cell[1] + dy

        dx, dy = DELTA[2]
        up_x, up_y = cell[0] + dx, cell[1] + dy

        dx, dy = DELTA[3]
        down_x, down_y = cell[0] + dx, cell[1] + dy

        if self.is_free(left_x, left_y) and self.S3_rewards[(left_x, left_y)] == -1:
            self._ripple((left_x, left_y), reward)

        if self.is_free(right_x, right_y) and self.S3_rewards[(right_x, right_y)] == -1:
            self._ripple((right_x, right_y), reward)

        if self.is_free(up_x, up_y) and self.S3_rewards[(up_x, up_y)] == -1:
            self._ripple((up_x, up_y), reward)

          
        if self.is_free(down_x, down_y) and self.S3_rewards[(down_x, down_y)] == -1:
            self._ripple((down_x, down_y), reward)