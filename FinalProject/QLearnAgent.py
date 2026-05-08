"""
Q LEARN AGENT
Please implement a Q-learning process which learns a feedback policy for
reaching the target position without hitting any obstacles. Similar to the previous task, you need to specify
a probability for exploration. You may use the same data structure for the feedback policy
"""
# imports
import numpy as np
import sys
from Agent import Agent
from MapCompressor import MapCompressor
from Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
class that implements the QLearn feedback policy for finding its way
to a target in the environment.
"""

class QLearnAgent(Agent):

  def __init__(self, 
               environment, 
               map_width=40,
               map_height=40,
               init_state=(0, 0),
               strategy_to_use="S1",
               learn_rate=0.5, 
               discount=0.95,  
               use_random_start=True,
               use_epsilon_decay=False,
               epsilon=1.0,
               min_epsilon=0.05,
               epsilon_decay_rate=0.0005
               ):
    super().__init__(environment, map_width, map_height, init_state, strategy_to_use) # init the agent, with the Q table
    self.learn_rate = learn_rate
    self.discount = discount
    self.use_random_start = use_random_start
    self.use_epsilon_decay = use_epsilon_decay
    self.epsilon = epsilon
    self.min_epsilon = min_epsilon
    self.epsilon_decay_rate = epsilon_decay_rate
    self.final_path = ([], []) # for plotting the path taken on the map
  
  # test the agent, meaning run it off it's Q table
  def test(self, steps=1000, init_state=(0, 0)):
    # set initial state
    curr_state = init_state

    # for plotting
    self.final_path[0].append(curr_state[0])
    self.final_path[1].append(curr_state[1])

    # have a limit on how long it can run for for sanity
    for step in range(steps):

      # epsilon is now 0 -> ALWAYS exploit, never explore
      action = self.exploit_or_explore(0, curr_state)

      # interact with the environment with this action A and current state S
      next_state, reward, done, crash = self.interact(curr_state, action, self.strategy)

      # for plotting
      self.final_path[0].append(next_state[0])
      self.final_path[1].append(next_state[1])

      # if DONE, like, hit the trigger
      if done: 
        print("AGENT SUCCESSFULLY FOUND TARGET!!! :>")
        break

      # if i hit a wall, DO BETTER
      if crash:
        print("AGENT CRASHED INTO BOUNDARY OR OBSTACLE! :(")
        break
      
      # no crash, not done
      curr_state = next_state


  # initiate QLearn algorithm to TRAIN the agent
  def train(self, episodes=1000, steps=1000, show_progress=True):

    # adding progress bar
    successes = 0
    crashes = 0
    timeouts = 0

    if show_progress:
      episode_iterator = tqdm(
        range(episodes),
        desc="Training QLearn",
        unit="episode"
      )
    else:
      episode_iterator = range(episodes)

    for episode in episode_iterator:
      # initialize S
      if self.use_random_start:
        curr_state = self.get_random_start_state() # set to a randomly selected point each episode
      else:
        curr_state = self.init_state # set to the given start point

      # if that mode is turned on: default is false
      if self.use_epsilon_decay:
        # use epsilon decay 
        epsilon = self.decay_epsilon(episode)
      else:
        # keep a constant epsilon
        epsilon = self.epsilon

      episode_complete = False

      # for each step of the episode
      for _ in range(steps):
        # choose next state/action by either exploitation or exploration
        action = self.exploit_or_explore(epsilon, curr_state)
        # interact with the environment with this action A and current state S
        next_state, reward, done, crash = self.interact(curr_state, action, self.strategy)
        # update the Q table
        self.update_Q_TABLE(curr_state, next_state, reward, action)
        # update the curr state to this one
        curr_state = next_state

    
        if done: # target
          successes += 1
          episode_complete = True
          break # stop, we reached terminal state

        if crash: # boundary
          crashes += 1
          episode_complete = True
          break
      
      if not episode_complete:
        timeouts += 1

      # end of episode
      if show_progress:
        episode_iterator.set_postfix({
          "successes": successes,
          "crashes": crashes,
          "timeouts": timeouts,
          "epsilon": f"{epsilon:.4f}"
        })


  # wrapper for specific update to Q table as directed by the 
  # QLearn algorithm.
  def update_Q_TABLE(self, curr_state: tuple, next_state: tuple, reward: float, action: int):
    # following variables are readability and writability.
    curr_x = curr_state[0]
    curr_y = curr_state[1]
    new_x  = next_state[0]
    new_y  = next_state[1]

    # update the Q table
    self.Q_TABLE[curr_x][curr_y][action] += self.learn_rate * (reward + (self.discount * np.max(self.Q_TABLE[new_x][new_y]) - self.Q_TABLE[curr_x][curr_y][action]))

  # use a greedy epsilon strategy to choose the next state
  # TODO: randomly break ties? for now, just exploit in ties
  def exploit_or_explore(self, epsilon: float, curr_state: tuple):
    
    p = np.random.uniform()
    action = None

    if p < epsilon:
      # epsilon chance to explore -- chose a random action
      action = np.random.randint(0, 4) # chose from l, r, u, d

    else:
      # 1 - epsilon chance to exploit -- choose based on Q table
      x = curr_state[0]
      y = curr_state[1]

      # randomly break ties, dumbass:
      q_vals = self.Q_TABLE[x][y]
      max_q = np.max(q_vals)
      actions = np.where(q_vals == max_q)[0]
      action = np.random.choice(actions)

    return action


  def get_random_start_state(self):
    """
    Optional helper for random-start training.
    """

    free_cells = self.environment.get_free_cells()
    state = free_cells[np.random.randint(len(free_cells))]

    while state == self.environment.target:
      state = free_cells[np.random.randint(len(free_cells))]

    return state

  # use exponential decay to slowly decay the exploration choice
  # not necessary, but appears beneficial to put more
  # emphasis on exploitation over time.
  def decay_epsilon(self, t: int):
    return self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * t)
  
  # plotting function, thank you marnieee
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
      title = f"Final Path of QLearn Agent, Start: {testing_init_point}, End: {target_point}, Strategy {self.strategy}"

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
# RUN THE AGENT ON ALL MAPS
# ==========================================================================================================

if __name__ == "__main__":
  np.random.seed(42) # woops
  learn_rate = 0.5
  epsilon = 0.0
  discount = 0.5
  episodes = 5000
  steps = 1000

  mc = MapCompressor()

  maps_to_test = [
    ("map1", mc.MAP_1_PATH),
    ("map2", mc.MAP_2_PATH),
    ("map3", mc.MAP_3_PATH),
    ("map4", mc.MAP_4_PATH)
  ]

  results = []

  strategy = "S1" # used below

  for map_name, map_path in maps_to_test:

    print("\n" + "=" * 80)
    print(f"RUNNING QLearn ON {map_name} with STRATEGY {strategy}")
    print(f"LEARN RATE = {learn_rate}, EPSILON = {epsilon}, DISCOUNT = {discount}")
    print("=" * 80)

    im = mc.compress(map_path)

    target_point = (im.shape[0] - 1, im.shape[1] - 1) # target stays consistent for the moment
    training_init_point = (0, 0) # start him from here when training
    testing_init_point = (0, 0) # just the same as before

    environment = Environment(im, target=target_point)

    print(f"{map_name} shape:", im.shape)
    print(f"{map_name} start:", testing_init_point)
    print(f"{map_name} target:", target_point)

    agent = QLearnAgent(
      environment,
      map_width=im.shape[0],
      map_height=im.shape[1],
      init_state=training_init_point,
      strategy_to_use=strategy,
      learn_rate=learn_rate,
      discount=discount,
      epsilon=epsilon,
    )

    agent.train(episodes=episodes, steps=steps)
    agent.test(steps=steps, init_state=testing_init_point)

    final_state = (agent.final_path[0][-1], agent.final_path[1][-1])
    path_length = len(agent.final_path[0])
    success = final_state == target_point

    print(f"{map_name} final path length:", path_length)
    print(f"{map_name} final state:", final_state)
    print(f"{map_name} target:", target_point)
    print(f"{map_name} success:", success)

    save_name = f"qlearn_final_path_{map_name}.png"

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