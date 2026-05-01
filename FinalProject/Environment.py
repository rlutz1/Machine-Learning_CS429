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

    def compute_reward(self, x: int, y: int, hit_boundary: bool,
                        strategy: str) -> float:
        if strategy == "S1":
            return self.reward_S1(x, y, hit_boundary)
        elif strategy == "S2":
            return self.reward_S2(x, y, hit_boundary)
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
        done: bool – True if the episode has ended (target reached, obstacle hit, max steps met)
        """
        x, y = state
        dx, dy = DELTA[action]
        nx, ny = x + dx, y + dy

        # Bounds checking
        if not self.in_bounds(nx, ny):
            reward = self.compute_reward(x, y, hit_boundary=True, strategy=strategy)
            # Agent stays in current cell and episode ends
            # return state, reward, True
            return state, reward, False # TODO: this may need to be true in training?

        # Obstacle
        if self.is_obstacle(nx, ny):
            reward = self.compute_reward(x, y, hit_boundary=True, strategy=strategy)
            # Treat obstacle collision same as boundary violation
            # return state, reward, True
            return state, reward, False # TODO: this may need to be true in training?

        # Target reached
        next_state = (nx, ny)
        if next_state == self.target:
            reward = self.compute_reward(nx, ny, hit_boundary=False, strategy=strategy)
            return next_state, reward, True  # End here we win (yay)

        # Normal steppin
        reward = self.compute_reward(nx, ny, hit_boundary=False, strategy=strategy)
        return next_state, reward, False

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