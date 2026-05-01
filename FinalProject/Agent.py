"""
BUILDING THE AGENT

You are required to write a Python class for the agent which keeps a
Q-table. At least a function for the interaction with an environment should be implemented. It performs an
interaction with an environment and updates the Q-table. As it is explained in the class, the Q-table should
be a 3-D matrix. S1 can be a naive strategy in which a very high reward is returned when the target position
is reached and a very negative reward is returned when an obstacle is reached
"""
# imports
import numpy as np
from MapCompressor import MapCompressor
from Environment import Environment

class Agent:

  def __init__(self, 
               environment, 
               map_width=40, 
               map_height=40,
               init_state=(0, 0),
               strategy_to_use="S1"):
    # make a 3d matrix: 
    # dimension 1: x coords (map_width)
    # dimension 2: y coords (map_height)
    # dimension 3: actions (U, D, L, R) (4)
    self.Q_TABLE = np.zeros((map_width, map_height, 4))
    self.environment = environment # environment to probe
    self.strategy = strategy_to_use # tie to the agent
    self.init_state=init_state
    # for not hardcoding every little thing.
    self.actions = {
      "L": 0, "R": 1, "U": 2, "D": 3
    }

  # interaction with environment simply a wrapper
  def interact(self, 
               curr_state: tuple,        # next coordinate (x, y)
               next_action: int,         # next action {0, 1, 2, 3} -> (LEFT / RIGHT / UP / DOWN)
               strategy: str = "S1" # strategy to use
               ):
    # probe the environment with your current state (S) and desired next action (A).
    # take action (A), observe next state (S') and reward (R)
    return self.environment.step(state=curr_state, action=next_action, strategy=strategy)

  # override with SARSA or QLearn?
  def train(self, episodes=1000):
    pass


class QLearnAgent(Agent):

  def __init__(self, 
               environment, 
               map_width=40,
               map_height=40,
               init_state=(0, 0),
               strategy_to_use="S1",
               learn_rate=0.5, 
               discount=0.5,  
               max_epsilon=1.0,
               min_epsilon=0.05,
               epsilon_decay_rate=0.0005
               ):
    super().__init__(environment, map_width, map_height, init_state, strategy_to_use) # init the agent, with the Q table
    self.learn_rate = learn_rate
    self.discount = discount
    self.max_epsilon = max_epsilon
    self.min_epsilon = min_epsilon
    self.epsilon_decay_rate = epsilon_decay_rate

  # initiate QLearn algorithm?
  def train(self, episodes=1000):

    for episode in range(episodes):
      # initialize S
      curr_state = self.init_state # set to the given start point
      # use epsilon decay 
      epsilon = self.decay_epsilon(episode)

      # for each step of the episode
      for step in range(0, 100):
        # choose next state/action by either exploitation or exploration
        action = self.exploit_or_explore(epsilon, curr_state)
        # interact with the environment with this action A and current state S
        next_state, reward, done = self.interact(curr_state, action, self.strategy)
        # if next state S' is not terminal
        if not done: 
          # update the Q table
          self.update_Q_TABLE(curr_state, next_state, reward, action)
          # update the curr state to this one
          curr_state = next_state
        else:
          break # stop, we reached terminal state


  def update_Q_TABLE(self, curr_state, next_state, reward, action):
    # following variables are readability
    curr_x = curr_state[0]
    curr_y = curr_state[1]
    new_x  = next_state[0]
    new_y  = next_state[1]

    QSA = self.Q_TABLE[curr_x][curr_y][action]
    max_QS_a_index = np.argmax(self.Q_TABLE[new_x][new_y])
    QS_a = self.Q_TABLE[new_x][new_y][max_QS_a_index]
    alpha = self.learn_rate
    R = reward
    gamma = self.discount

    # update
    self.Q_TABLE[curr_x][curr_y][action] = QSA + alpha * (R + (gamma * QS_a) - QSA) 



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
      action = np.argmax(self.Q_TABLE[x][y])

    return action


  # use exponential decay to slowly decay the exploration choice
  # not necessary, but appears beneficial to put more
  # emphasis on exploitation over time.
  def decay_epsilon(self, t):
    return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * t)

    # next_state, reward, done = self.interact()
   
 

# ==========================================================================================================
# TESTING
# ==========================================================================================================

mc = MapCompressor()
im = mc.compress(mc.MAP_1_PATH)

environment = Environment(im, (0, 0)) # testing only

# agent = Agent(environment, im.shape[0], im.shape[1]) # agent 
agent = QLearnAgent(environment, im.shape[0], im.shape[1]) # agent 
agent.train(episodes=1000)
print(agent.Q_TABLE)
# plt.imshow(im)
# plt.colorbar()
# plt.show()