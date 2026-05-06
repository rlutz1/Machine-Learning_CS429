from Evaluate import exp1_map_complexity
from Evaluate import SEED
from Evaluate import exp3_discount_value
from Evaluate import exp4_reward_strategy

import numpy as np

# SEED = 42
# EPISODES = 5000 # training episodes per trial
# STEPS = 1000 # max steps per episode during training
# LEARN_RATE = 0.5

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # r3 = exp3_discount_value(epsilon=0.2, strategy="S1")
   
    # eps = 0.2, disc = 0.5
    # r4 = exp4_reward_strategy(best_epsilon=0.2, best_discount=0.5)
   
    # eps = 0.2, disc = 0.95
    r4 = exp4_reward_strategy(best_epsilon=0.2, best_discount=0.95)
    # r4 = exp4_reward_strategy(best_epsilon=0.2, best_discount=0.95)
   
    # eps = 0.5, disc = 0.5
    # r4 = exp4_reward_strategy(best_epsilon=0.5, best_discount=0.5)
   
    # THIS IS TO BE RUN WITH DECAY TURNED TRUE IN EVAL
    # r4 = exp4_reward_strategy(best_epsilon=1.0, best_discount=0.95)