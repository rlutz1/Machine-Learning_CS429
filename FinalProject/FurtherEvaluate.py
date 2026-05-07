from Evaluate import exp1_map_complexity
from Evaluate import exp2_exploration_rate

from Evaluate import exp3_discount_value
from Evaluate import exp4_reward_strategy

import numpy as np

# SEED = 42
# EPISODES = 5000 # training episodes per trial
# STEPS = 1000 # max steps per episode during training
# LEARN_RATE = 0.5

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # # to set eps lower and see what a higher discount does
    # r3 = exp3_discount_value(epsilon=0.2, strategy="S1")

    #  # to set eps lower and see what a higher discount does
    # r3_2 = exp2_exploration_rate(discount=0.2, strategy="S1")
   
    # to set eps lower and see what a higher discount does
    # r3 = exp3_discount_value(epsilon=0.1, strategy="S1")



    # from the table:
    # eps = 0, disc = 0.5
    # r4 = exp4_reward_strategy(best_epsilon=0, best_discount=0.5)

    """
    ========================================================================================
    Exp 4 – Reward Strategy  [eps=0, gamma=0.5]
    ========================================================================================
    agent      strategy   epsilon    discount   time_s     episodes   accuracy   avg_path   
    ----------------------------------------------------------------------------------------
    SARSA      S1         0          0.5        85.63      5000       99.8       45.5       
    QLearn     S1         0          0.5        36.1       5000       100.0      46.4       
    SARSA      S2         0          0.5        31.02      5000       76.2       32.7       
    QLearn     S2         0          0.5        27.93      5000       88.2       50.0       
    SARSA      S3         0          0.5        332.5      5000       0.0        N/A        
    QLearn     S3         0          0.5        112.18     5000       0.5        3.2        
    ========================================================================================
    
    """


    # eps = 0.5, disc = 0.1
    r4 = exp4_reward_strategy(best_epsilon=0.5, best_discount=0.1)

    """
    ========================================================================================
    Exp 4 – Reward Strategy  [eps=0.5, gamma=0.1]
    ========================================================================================
    agent      strategy   epsilon    discount   time_s     episodes   accuracy   avg_path   
    ----------------------------------------------------------------------------------------
    SARSA      S1         0.5        0.1        42.02      5000       100.0      39.9       
    QLearn     S1         0.5        0.1        20.64      5000       75.8       175.6      
    SARSA      S2         0.5        0.1        94.33      5000       13.0       10.9       
    QLearn     S2         0.5        0.1        20.16      5000       12.8       11.1       
    SARSA      S3         0.5        0.1        77.79      5000       34.7       18.3       
    QLearn     S3         0.5        0.1        20.18      5000       21.2       15.7       
    ========================================================================================
    """
   
    # eps = 0.2, disc = 0.95
    # r4 = exp4_reward_strategy(best_epsilon=0.2, best_discount=0.95)
    # run as well so that ensure random fix is all gucci
    # r4 = exp4_reward_strategy(best_epsilon=0.2, best_discount=0.95)
   
    # eps = 0.5, disc = 0.5
    # r4 = exp4_reward_strategy(best_epsilon=0.5, best_discount=0.5)
   
    # THIS IS TO BE RUN WITH DECAY TURNED TRUE IN EVAL
    # r4 = exp4_reward_strategy(best_epsilon=1.0, best_discount=0.95)