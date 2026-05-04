"""
SARSA AGENT

Implements SARSA for learning a feedback policy to reach a target
without hitting obstacles or boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from tqdm import tqdm

from Agent import Agent
from MapCompressor import MapCompressor
from Environment import Environment


class SARSAAgent(Agent):

  def __init__(self,
               environment,
               map_width=40,
               map_height=40,
               init_state=(0, 0),
               strategy_to_use="S1",
               learn_rate=0.5,
               discount=0.99,
               use_epsilon_decay=True,
               epsilon=1.0,
               min_epsilon=0.01,
               epsilon_decay_rate=0.00005,
               use_random_start=False,
               use_reward_shaping=True):

    super().__init__(
      environment,
      map_width,
      map_height,
      init_state,
      strategy_to_use
    )

    self.learn_rate = learn_rate
    self.discount = discount

    self.use_epsilon_decay = use_epsilon_decay
    self.epsilon = epsilon
    self.min_epsilon = min_epsilon
    self.epsilon_decay_rate = epsilon_decay_rate

    self.use_random_start = use_random_start
    self.use_reward_shaping = use_reward_shaping

    self.distance_to_target = self.compute_distance_to_target()

    self.final_path = ([], [])


  def compute_distance_to_target(self):
    """
    Compute shortest valid grid distance from every free cell to the target
    using BFS. Obstacles and boundaries are unreachable.
    """

    height = self.environment.n_rows
    width = self.environment.n_cols

    distances = np.full((width, height), np.inf)

    tx, ty = self.environment.target

    queue = deque()
    queue.append((tx, ty))
    distances[tx][ty] = 0

    deltas = [
      (-1, 0),  # left
      (1, 0),   # right
      (0, -1),  # up
      (0, 1)    # down
    ]

    while queue:
      x, y = queue.popleft()

      for dx, dy in deltas:
        nx = x + dx
        ny = y + dy

        if self.environment.in_bounds(nx, ny) and not self.environment.is_obstacle(nx, ny):
          if distances[nx][ny] == np.inf:
            distances[nx][ny] = distances[x][y] + 1
            queue.append((nx, ny))

    return distances


  def get_random_start_state(self):
    """
    Optional helper for random-start training.
    """

    free_cells = self.environment.get_free_cells()
    state = free_cells[np.random.randint(len(free_cells))]

    while state == self.environment.target:
      state = free_cells[np.random.randint(len(free_cells))]

    return state


  def get_valid_actions(self, state: tuple):
    """
    Return only actions that stay in bounds and do not hit an obstacle.
    """

    x, y = state

    deltas = {
      0: (-1, 0),  # LEFT
      1: (1, 0),   # RIGHT
      2: (0, -1),  # UP
      3: (0, 1)    # DOWN
    }

    valid_actions = []

    for action, (dx, dy) in deltas.items():
      nx = x + dx
      ny = y + dy

      if self.environment.in_bounds(nx, ny) and not self.environment.is_obstacle(nx, ny):
        valid_actions.append(action)

    return valid_actions


  def shape_reward(self, curr_state, next_state, reward):
    """
    Reward moves that reduce true map distance to the target.
    """

    if not self.use_reward_shaping:
      return reward

    curr_x, curr_y = curr_state
    next_x, next_y = next_state

    curr_dist = self.distance_to_target[curr_x][curr_y]
    next_dist = self.distance_to_target[next_x][next_y]

    if curr_dist == np.inf or next_dist == np.inf:
      return reward

    if next_dist < curr_dist:
      reward += 5.0
    elif next_dist > curr_dist:
      reward -= 5.0
    else:
      reward -= 0.5

    return reward


  def train(self, episodes=1000, steps=1000, show_progress=True):

    successes = 0
    crashes = 0
    timeouts = 0

    if show_progress:
      episode_iterator = tqdm(
        range(episodes),
        desc="Training SARSA",
        unit="episode"
      )
    else:
      episode_iterator = range(episodes)

    for episode in episode_iterator:

      if self.use_random_start:
        curr_state = self.get_random_start_state()
      else:
        curr_state = self.init_state

      if self.use_epsilon_decay:
        epsilon = self.decay_epsilon(episode)
      else:
        epsilon = self.epsilon

      curr_action = self.exploit_or_explore(epsilon, curr_state)

      episode_ended = False

      for _ in range(steps):

        next_state, reward, done, crash = self.interact(
          curr_state,
          curr_action,
          self.strategy
        )

        if self.use_reward_shaping:
          reward = self.shape_reward(curr_state, next_state, reward)

        if done or crash:
          self.update_Q_TABLE(
            curr_state,
            curr_action,
            reward,
            next_state,
            next_action=None,
            terminal=True
          )

          if done:
            successes += 1
          if crash:
            crashes += 1

          episode_ended = True
          break

        next_action = self.exploit_or_explore(epsilon, next_state)

        self.update_Q_TABLE(
          curr_state,
          curr_action,
          reward,
          next_state,
          next_action,
          terminal=False
        )

        curr_state = next_state
        curr_action = next_action

      if not episode_ended:
        timeouts += 1

      if show_progress:
        episode_iterator.set_postfix({
          "successes": successes,
          "crashes": crashes,
          "timeouts": timeouts,
          "epsilon": f"{epsilon:.4f}"
        })


  def update_Q_TABLE(self,
                     curr_state: tuple,
                     curr_action: int,
                     reward: float,
                     next_state: tuple,
                     next_action: int = None,
                     terminal: bool = False):

    curr_x = curr_state[0]
    curr_y = curr_state[1]

    old_q = self.Q_TABLE[curr_x][curr_y][curr_action]

    if terminal:
      target = reward
    else:
      next_x = next_state[0]
      next_y = next_state[1]
      next_q = self.Q_TABLE[next_x][next_y][next_action]
      target = reward + self.discount * next_q

    self.Q_TABLE[curr_x][curr_y][curr_action] = old_q + self.learn_rate * (
      target - old_q
    )


  def exploit_or_explore(self, epsilon: float, curr_state: tuple):
    """
    Epsilon-greedy over valid actions.
    """

    valid_actions = self.get_valid_actions(curr_state)

    if len(valid_actions) == 0:
      return np.random.randint(0, 4)

    p = np.random.uniform()

    if p < epsilon:
      return int(np.random.choice(valid_actions))

    x = curr_state[0]
    y = curr_state[1]

    q_vals = self.Q_TABLE[x][y]

    valid_q_vals = [q_vals[a] for a in valid_actions]
    max_q = np.max(valid_q_vals)

    best_actions = [a for a in valid_actions if q_vals[a] == max_q]

    return int(np.random.choice(best_actions))


  def greedy_action(self, curr_state: tuple):
    """
    Deterministic valid greedy action for testing.
    """

    valid_actions = self.get_valid_actions(curr_state)

    if len(valid_actions) == 0:
      x, y = curr_state
      return int(np.argmax(self.Q_TABLE[x][y]))

    x, y = curr_state
    q_vals = self.Q_TABLE[x][y]

    best_action = valid_actions[0]
    best_q = q_vals[best_action]

    for action in valid_actions:
      if q_vals[action] > best_q:
        best_q = q_vals[action]
        best_action = action

    return int(best_action)


  def decay_epsilon(self, t: int):
    return self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(
      -self.epsilon_decay_rate * t
    )


  def test(self, steps=1000, init_state=(0, 0)):

    curr_state = init_state

    self.final_path = ([], [])
    visited_states = set()

    self.final_path[0].append(curr_state[0])
    self.final_path[1].append(curr_state[1])

    for _ in range(steps):

      # if curr_state in visited_states:
      #   print("AGENT GOT STUCK IN A LOOP DURING TESTING.")
      #   break

      visited_states.add(curr_state)

      action = self.greedy_action(curr_state)

      next_state, reward, done, crash = self.interact(
        curr_state,
        action,
        self.strategy
      )

      self.final_path[0].append(next_state[0])
      self.final_path[1].append(next_state[1])

      if done:
        print("AGENT SUCCESSFULLY FOUND TARGET!!! :>")
        break

      if crash:
        print("AGENT CRASHED INTO BOUNDARY OR OBSTACLE! :(")
        break

      curr_state = next_state


  def plot_path(self,
                map_abstraction,
                target_point,
                testing_init_point=(0, 0),
                title=None,
                save_name="sarsa_final_path.png"):

    if len(self.final_path[0]) == 0:
      print("No path to plot. Run test() before calling plot_path().")
      return

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.imshow(map_abstraction, cmap="gray", origin="upper")

    if title is None:
      title = f"Final Path of SARSA Agent, Start: {testing_init_point}, End: {target_point}"

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Map Width / x-coordinate")
    ax.set_ylabel("Map Height / y-coordinate")

    height = map_abstraction.shape[0]
    width = map_abstraction.shape[1]

    ax.set_xticks(np.arange(0, width, 4))
    ax.set_yticks(np.arange(0, height, 4))

    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)

    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.25, alpha=0.35)
    ax.grid(which="major", color="gray", linestyle="-", linewidth=0.7, alpha=0.7)

    path_x = self.final_path[0]
    path_y = self.final_path[1]

    ax.plot(
      path_x,
      path_y,
      color="blue",
      linewidth=1.5,
      alpha=0.8,
      label="Path Line",
      zorder=3
    )

    ax.scatter(
      path_x[0],
      path_y[0],
      c="green",
      s=80,
      edgecolors="black",
      linewidths=0.7,
      label="Start",
      zorder=5
    )

    if len(path_x) > 2:
      ax.scatter(
        path_x[1:-1],
        path_y[1:-1],
        c="blue",
        s=25,
        alpha=0.85,
        label="Transit",
        zorder=4
      )

    ax.scatter(
      path_x[-1],
      path_y[-1],
      c="red",
      s=80,
      edgecolors="black",
      linewidths=0.7,
      label="End",
      zorder=6
    )

    ax.scatter(
      target_point[0],
      target_point[1],
      c="yellow",
      marker="*",
      s=250,
      edgecolors="black",
      linewidths=0.7,
      label="Target",
      zorder=7
    )

    ax.annotate(
      f"Start {testing_init_point}",
      xy=(path_x[0], path_y[0]),
      xytext=(path_x[0] + 1, path_y[0] + 1),
      fontsize=8,
      arrowprops=dict(arrowstyle="->", linewidth=0.8)
    )

    ax.annotate(
      f"End ({path_x[-1]}, {path_y[-1]})",
      xy=(path_x[-1], path_y[-1]),
      xytext=(path_x[-1] + 1, path_y[-1] + 1),
      fontsize=8,
      arrowprops=dict(arrowstyle="->", linewidth=0.8)
    )

    ax.annotate(
      f"Target {target_point}",
      xy=(target_point[0], target_point[1]),
      xytext=(target_point[0] - 10, target_point[1] - 2),
      fontsize=8,
      arrowprops=dict(arrowstyle="->", linewidth=0.8)
    )

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    fig.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {save_name}")


# ==========================================================================================================
# TESTING ALL MAPS
# ==========================================================================================================

if __name__ == "__main__":

  np.random.seed(42)

  mc = MapCompressor()

  maps_to_test = [
    # ("map1", mc.MAP_1_PATH),
    ("map2", mc.MAP_2_PATH),
    # ("map3", mc.MAP_3_PATH),
    ("map4", mc.MAP_4_PATH)
  ]

  def choose_target(environment):
    """
    Try to use the bottom-right corner as the target.
    If that is not free, use the closest free cell to the bottom-right corner.
    """

    bottom_right = (environment.n_cols - 1, environment.n_rows - 1)

    if environment.is_free(bottom_right[0], bottom_right[1]):
      return bottom_right

    free_cells = environment.get_free_cells()

    target = min(
      free_cells,
      key=lambda p: abs(p[0] - bottom_right[0]) + abs(p[1] - bottom_right[1])
    )

    return target


  def choose_start(environment):
    """
    Try to use (2, 1) as the start.
    If that is not valid for a map, use the closest free cell to (2, 1).
    """

    # preferred_start = (2, 1)
    preferred_start = (0, 0)

    if environment.is_free(preferred_start[0], preferred_start[1]):
      return preferred_start

    free_cells = environment.get_free_cells()

    start = min(
      free_cells,
      key=lambda p: abs(p[0] - preferred_start[0]) + abs(p[1] - preferred_start[1])
    )

    return start


  results = []

  for map_name, map_path in maps_to_test:

    print("\n" + "=" * 80)
    print(f"RUNNING SARSA ON {map_name}")
    print("=" * 80)

    im = mc.compress(map_path)

    # Temporary target only so we can create the environment object.
    # We choose a valid target immediately after.
    temp_target = (0, 0)

    temp_environment = Environment(im, target=temp_target)

    target_point = choose_target(temp_environment)

    environment = Environment(im, target=target_point)

    training_init_point = choose_start(environment)
    testing_init_point = training_init_point

    print(f"{map_name} shape:", im.shape)
    print(f"{map_name} start:", testing_init_point)
    print(f"{map_name} target:", target_point)

    agent = SARSAAgent(
      environment,
      map_width=im.shape[1],
      map_height=im.shape[0],
      init_state=training_init_point,
      strategy_to_use="S2",
      learn_rate=0.7,
      discount=0.95,
      epsilon=0.2,
      # min_epsilon=0.01,
      # epsilon_decay_rate=0.00005,
      use_epsilon_decay=False,
      use_random_start=False,
      use_reward_shaping=False
    )

    agent.train(
      episodes=20000,
      steps=1000,
      show_progress=True
    )

    agent.test(
      steps=10000,
      init_state=testing_init_point
    )

    final_state = (agent.final_path[0][-1], agent.final_path[1][-1])
    path_length = len(agent.final_path[0])
    success = final_state == target_point

    print(f"{map_name} final path length:", path_length)
    print(f"{map_name} final state:", final_state)
    print(f"{map_name} target:", target_point)
    print(f"{map_name} success:", success)

    save_name = f"sarsa_final_path_{map_name}.png"

    agent.plot_path(
      map_abstraction=im,
      target_point=target_point,
      testing_init_point=testing_init_point,
      save_name=save_name
    )

    results.append({
      "map": map_name,
      "success": success,
      "path_length": path_length,
      "final_state": final_state,
      "target": target_point,
      "start": testing_init_point,
      "plot": save_name
    })

  print("\n" + "=" * 80)
  print("SUMMARY")
  print("=" * 80)

  for result in results:
    print(
      f"{result['map']}: "
      f"success={result['success']}, "
      f"start={result['start']}, "
      f"target={result['target']}, "
      f"path_length={result['path_length']}, "
      f"final_state={result['final_state']}, "
      f"plot={result['plot']}"
    )