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
               learn_rate=0.7, 
               discount=0.95,  
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

  
  # test the agent, meaning run it off it's Q table
  def test(self, steps=1000):

    # set initial state
    curr_state = self.init_state

    # have a limit on how long it can run for for sanity
    for step in range(steps):
      print(curr_state)
      # epsilon is now 0 -> ALWAYS exploit
      action = self.exploit_or_explore(0, curr_state)

      # interact with the environment with this action A and current state S
      next_state, reward, done, crash = self.interact(curr_state, action, self.strategy)

      # if DONE, like, hit the trigger
      if done: 
        print("AGENT SUCCESSFULLY FOUND TARGET!!! :>")
        break
      elif crash:
        print("AGENT CRASHED INTO BOUNDARY OR OBSTACLE! :(")
        break
      else: # no crash, not done
        curr_state = next_state



  # initiate QLearn algorithm to TRAIN the agent
  def train(self, episodes=1000):

    for episode in range(episodes):
      # initialize S
      curr_state = self.init_state # set to the given start point
      # use epsilon decay 
      epsilon = self.decay_epsilon(episode)
      

      # for each step of the episode
      for _ in range(1000):
        # choose next state/action by either exploitation or exploration
        action = self.exploit_or_explore(epsilon, curr_state)
        # print(f"next action {action}")
        # interact with the environment with this action A and current state S
        next_state, reward, done, crash = self.interact(curr_state, action, self.strategy)
        # print(f"next state {next_state}, reward {reward}")

        # if next state S' is not target
        # or the boundary
        if not done: 
          # update the Q table
          self.update_Q_TABLE(curr_state, next_state, reward, action)
          # update the curr state to this one
          curr_state = next_state
        else: # boundary or target
          # print("??")
          break # stop, we reached terminal state

  # wrapper for specific update to Q table as directed by the 
  # QLearn algorithm.
  def update_Q_TABLE(self, curr_state, next_state, reward, action):
    # following variables are readability and writability.
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

    # update according to QLearn
    self.Q_TABLE[curr_x][curr_y][action] = QSA + alpha * (R + (gamma * QS_a) - QSA) 
    # print(self.Q_TABLE[curr_x][curr_y][action])
    # print(f"QSA {QSA}, QS_a {QS_a}, R {R}")

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
   
 

# ==========================================================================================================
# TESTING
# ==========================================================================================================

mc = MapCompressor()
im = mc.compress(mc.MAP_3_PATH)

target_point = (39, 39)
init_point = (10, 20)

environment = Environment(im, target=target_point) # testing only

# agent = Agent(environment, im.shape[0], im.shape[1]) # agent 
agent = QLearnAgent(
  environment, 
  map_width=im.shape[0],
  map_height=im.shape[1],
  init_state=init_point,
  strategy_to_use="S2",
  learn_rate=0.7, 
  discount=0.95,  
  max_epsilon=1.0,
  min_epsilon=0.05,
  epsilon_decay_rate=0.0005
  )

agent.train(episodes=10000)

agent.test(steps=10000)


np.set_printoptions(threshold=sys.maxsize)
print(agent.Q_TABLE)
# plt.imshow(im)
# plt.colorbar()
# plt.show()