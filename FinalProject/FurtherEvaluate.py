from Evaluate import exp1_map_complexity
from Evaluate import exp2_exploration_rate

from Evaluate import exp3_discount_value
from Evaluate import exp4_reward_strategy

import numpy as np

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # # to set eps lower and see what a higher discount does
    # r3 = exp3_discount_value(epsilon=0.2, strategy="S1")

    #  # to set eps lower and see what a higher discount does
    # r3_2 = exp2_exploration_rate(discount=0.2, strategy="S1")
   
    # to set eps lower and see what a higher discount does
    # r3 = exp3_discount_value(epsilon=0.1, strategy="S1")

    # ============================================================
    # data for the report in case useful: 
    # ============================================================

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

    # # eps = 0.5, disc = 0.1
    # r4 = exp4_reward_strategy(best_epsilon=0.5, best_discount=0.1)
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

    # empirically a decent balance
    # eps = 0.2, disc = 0.95
    # r4 = exp4_reward_strategy(best_epsilon=0.2, best_discount=0.95)
    
    # r1 = exp1_map_complexity(epsilon=1, discount=0.95)
    """
    ==================================================================
    Exp 1 – Map Complexity  [eps=1, gamma=0.95, S1]
    ==================================================================
    map        agent      time_s     episodes   accuracy   avg_path   
    ------------------------------------------------------------------
    map1       SARSA      20.62      5000       65.4       15.5       
    map1       QLearn     23.19      5000       100.0      17.4       
    map2       SARSA      130.32     5000       100.0      37.1       
    map2       QLearn     83.27      5000       99.4       70.7       
    map3       SARSA      136.78     5000       99.9       41.3       
    map3       QLearn     83.11      5000       100.0      71.4       
    map4       SARSA      63.32      5000       100.0      50.3       
    map4       QLearn     36.32      5000       84.5       192.6      
    ==================================================================

    ========================================================================================
    Exp 4 – Reward Strategy  [eps=0.2, gamma=0.95]
    ========================================================================================
    agent      strategy   epsilon    discount   time_s     episodes   accuracy   avg_path   
    ----------------------------------------------------------------------------------------
    SARSA      S1         0.2        0.95       84.36      5000       94.5       42.5       
    QLearn     S1         0.2        0.95       107.9      5000       78.9       171.6      
    SARSA      S2         0.2        0.95       155.77     5000       93.1       37.8       
    QLearn     S2         0.2        0.95       108.72     5000       71.0       30.8       
    SARSA      S3         0.2        0.95       108.01     5000       94.9       38.8       
    QLearn     S3         0.2        0.95       83.15      5000       83.9       36.9       
    ========================================================================================
    """

    # decay
    # THIS IS TO BE RUN WITH DECAY TURNED TRUE IN EVAL
    # r4 = exp4_reward_strategy(best_epsilon=1.0, best_discount=0.95)
    """
    ========================================================================================
    Exp 4 – Reward Strategy  [eps=1.0, gamma=0.95]
    ========================================================================================
    agent      strategy   epsilon    discount   time_s     episodes   accuracy   avg_path   
    ----------------------------------------------------------------------------------------
    SARSA      S1         1.0        0.95       46.5       5000       100.0      50.3       
    QLearn     S1         1.0        0.95       23.13      5000       84.5       192.6      
    SARSA      S2         1.0        0.95       45.22      5000       95.5       38.6       
    QLearn     S2         1.0        0.95       40.67      5000       50.7       24.6       
    SARSA      S3         1.0        0.95       38.16      5000       90.8       39.2       
    QLearn     S3         1.0        0.95       22.27      5000       67.8       30.3       
    ========================================================================================
    """
    # note to self: ran above with False and did indeed get diff numbers to confirm
    # that decay was in fact actually running.

    # would high eps and low discount work?
    # my guess: not really--encouraging too much aimless wondering and not using the prior knowledge enough.
    # r4 = exp4_reward_strategy(best_epsilon=0.95, best_discount=0.2)
    """
    evidence does back up the guess. too much wondering and severe drop in accuracy
    ========================================================================================
    Exp 4 – Reward Strategy  [eps=0.95, gamma=0.2]
    ========================================================================================
    agent      strategy   epsilon    discount   time_s     episodes   accuracy   avg_path   
    ----------------------------------------------------------------------------------------
    SARSA      S1         0.95       0.2        380.83     5000       100.0      39.4       
    QLearn     S1         0.95       0.2        83.64      5000       82.1       156.9      
    SARSA      S2         0.95       0.2        413.37     5000       15.7       12.0       
    QLearn     S2         0.95       0.2        22.34      5000       13.2       11.4       
    SARSA      S3         0.95       0.2        117.06     5000       37.5       19.1       
    QLearn     S3         0.95       0.2        23.72      5000       23.4       15.3       
    ========================================================================================
    """

    # what if about the same
    # r4 = exp4_reward_strategy(best_epsilon=0.5, best_discount=0.5)
    """
    same also pretty bad
    ========================================================================================
    Exp 4 – Reward Strategy  [eps=0.5, gamma=0.5]
    ========================================================================================
    agent      strategy   epsilon    discount   time_s     episodes   accuracy   avg_path   
    ----------------------------------------------------------------------------------------
    SARSA      S1         0.5        0.5        119.0      5000       99.7       39.4       
    QLearn     S1         0.5        0.5        39.21      5000       73.4       178.3      
    SARSA      S2         0.5        0.5        192.88     5000       22.6       21.8       
    QLearn     S2         0.5        0.5        77.57      5000       23.1       15.1       
    SARSA      S3         0.5        0.5        145.66     5000       72.8       31.6       
    QLearn     S3         0.5        0.5        84.43      5000       30.6       17.5       
    ========================================================================================
    """
    # lower both a little bit
    # r4 = exp4_reward_strategy(best_epsilon=0.1, best_discount=0.8)

    """
    intruiguing--back up a bit
    ========================================================================================
    Exp 4 – Reward Strategy  [eps=0.1, gamma=0.8]
    ========================================================================================
    agent      strategy   epsilon    discount   time_s     episodes   accuracy   avg_path   
    ----------------------------------------------------------------------------------------
    SARSA      S1         0.1        0.8        172.94     5000       100.0      42.3       
    QLearn     S1         0.1        0.8        104.79     5000       100.0      70.2       
    SARSA      S2         0.1        0.8        104.87     5000       88.5       36.1       
    QLearn     S2         0.1        0.8        27.7       5000       70.6       30.9       
    SARSA      S3         0.1        0.8        42.76      5000       77.3       32.6       
    QLearn     S3         0.1        0.8        28.31      5000       71.8       32.7       
    ========================================================================================
    """

    # lower both a little bit
    # r4 = exp4_reward_strategy(best_epsilon=0.0, best_discount=0.8)
    """
    0 epsilon does BAD in this reward strat lol
    ========================================================================================
    Exp 4 – Reward Strategy  [eps=0.0, gamma=0.8]
    ========================================================================================
    agent      strategy   epsilon    discount   time_s     episodes   accuracy   avg_path   
    ----------------------------------------------------------------------------------------
    SARSA      S1         0.0        0.8        78.68      5000       99.8       45.5       
    QLearn     S1         0.0        0.8        113.59     5000       100.0      46.4       
    SARSA      S2         0.0        0.8        116.27     5000       87.8       35.6       
    QLearn     S2         0.0        0.8        101.56     5000       86.8       35.5       
    SARSA      S3         0.0        0.8        619.21     5000       0.3        2.0        
    QLearn     S3         0.0        0.8        323.79     5000       0.3        2.5        
    ========================================================================================
    """
   
   