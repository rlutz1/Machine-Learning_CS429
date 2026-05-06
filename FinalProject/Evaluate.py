"""
Evaluate.py
Performance evaluation comparing SARSA and Q-Learning.

Experiments:
  1. Map Complexity – both agents on all 4 maps, fixed hyperparameters
  2. Exploration Rate – both agents on Map 4, gamma=0.5, epsilon in {0, 0.5, 1}
  3. Discount Value – both agents on Map 4, epsilon=0.5, gamma in {0.1, 0.5, 1}
  4. Reward Strategy – both agents on Map 4, best params from 2 & 3, strategies S1/S2/S3

Metrics collected per trial:
  - Training time
  - Episodes trained
  - Test accuracy (% of all free starting cells that reach the target)
  - Average path length (over successful runs)
"""

import time
import numpy as np
from tqdm import tqdm

from MapCompressor import MapCompressor
from Environment import Environment
from SarsaAgent import SARSAAgent
from QLearnAgent import QLearnAgent


SEED = 42
EPISODES = 5000 # training episodes per trial
STEPS = 1000 # max steps per episode during training
LEARN_RATE = 0.5 # learning rate for consistency between agents


# ── helpers ───────────────────────────────────────────────────────────────────

def find_target(grid):
    """
    Returns the bottom-right free cell on the grid,
    or the nearest free cell to the bottom-right if that corner is blocked.
    """
    rows, cols = grid.shape
    br_x, br_y = cols - 1, rows - 1

    # Check corner first
    if grid[br_y, br_x] >= 128:
        return (br_x, br_y)

    # Otherwise scan for the nearest free cell to the corner
    best, best_dist = None, float("inf") # track nearest free cell
    for y in range(rows):
        for x in range(cols):
            if grid[y, x] >= 128:
                d = abs(x - br_x) + abs(y - br_y)
                if d < best_dist:
                    best_dist = d
                    best = (x, y)
    return best


def greedy_rollout(agent, env, start, max_steps):
    """
    Run the learned policy from start until done, crash, or max_steps.
    Returns (reached_target: bool, steps_taken: int).
    """
    state = start

    for step in range(max_steps):
        # SARSA uses greedy_action which filters to valid actions
        # QLearn uses exploit_or_explore with epsilon=0 (always greedy)
        if isinstance(agent, SARSAAgent):
            action = agent.greedy_action(state)
        else:
            action = agent.exploit_or_explore(0, state)

        next_state, reward, done, crash = agent.interact(state, action, agent.strategy)

        if done:
            return True, step + 1
        if crash:
            return False, step + 1

        state = next_state

    return False, max_steps  # timeout counts as failure


def test_accuracy(agent, env):
    """
    Evaluate the trained policy from every free cell (excluding the target).
    Returns (accuracy_pct: float, avg_path_len: float or "N/A").
    """
    free_cells = env.get_free_cells()
    starts = [c for c in free_cells if c != env.target]

    # Give the agent enough steps to find any reasonable path not 10k like someone said
    max_steps = env.n_rows * env.n_cols

    successes = 0
    path_lengths = []

    for start in starts:
        reached, length = greedy_rollout(agent, env, start, max_steps)
        if reached:
            successes += 1
            path_lengths.append(length)

    accuracy = 100.0 * successes / len(starts) if starts else 0.0
    avg_len = round(sum(path_lengths) / len(path_lengths), 1) if path_lengths else "N/A"

    return round(accuracy, 1), avg_len


def run_trial(grid, target, agent_type, episodes, steps, epsilon, discount, strategy):
    """
    Create an agent, train it, evaluate it, and return a metrics dict.

    grid : compressed map numpy array
    target : (x, y) target position
    agent_type : "SARSA" or "QLearn"
    """
    env = Environment(grid, target)
    rows, cols = grid.shape

    # Common constructor args for both agent types
    common = dict(
        environment=env,
        map_width=cols,
        map_height=rows,
        strategy_to_use=strategy,
        learn_rate=LEARN_RATE,
        epsilon=epsilon,
        discount=discount,
        use_random_start=True,
        use_epsilon_decay=False # TODO
    )

    if agent_type == "SARSA":
        agent = SARSAAgent(**common, use_reward_shaping=False) 
    else:
        agent = QLearnAgent(**common) 

    np.random.seed(SEED) # reset seed prior to training?

    # Train and time it
    t0 = time.time()
    agent.train(episodes=episodes, steps=steps, show_progress=False)
    elapsed = round(time.time() - t0, 2)

    # Evaluate from all starting positions
    accuracy, avg_path = test_accuracy(agent, env)

    return {
        "agent": agent_type,
        "strategy": strategy,
        "epsilon": epsilon,
        "discount": discount,
        "episodes": episodes,
        "time_s": elapsed,
        "accuracy": accuracy,
        "avg_path": avg_path,
    }


def print_table(rows, col_keys, title, col_width=11):
    """Print results as a fixed-width table."""
    w = col_width
    sep = "=" * (w * len(col_keys))
    print(f"\n{sep}")
    print(title)
    print(sep)
    print("".join(k.ljust(w) for k in col_keys))
    print("-" * (w * len(col_keys)))
    for row in rows:
        print("".join(str(row.get(k, "")).ljust(w) for k in col_keys))
    print(sep)


def best_result(results):
    """Pick the result with the highest accuracy, then shortest path as tiebreaker."""
    def score(r):
        path = r["avg_path"]
        path_score = -float(path) if isinstance(path, (int, float)) else 0.0
        return (r["accuracy"], path_score)
    return max(results, key=score)


# ── map setup ─────────────────────────────────────────────────────────────────

mc = MapCompressor()

MAP_PATHS = [
    ("map1", mc.MAP_1_PATH),
    ("map2", mc.MAP_2_PATH),
    ("map3", mc.MAP_3_PATH),
    ("map4", mc.MAP_4_PATH),
]


# ── experiments ───────────────────────────────────────────────────────────────

def exp1_map_complexity(episodes=EPISODES, steps=STEPS,
                         epsilon=0.2, discount=0.95, strategy="S1"):
    """
    Experiment 1: Map Complexity
    Run both agents on all 4 maps with the same fixed hyperparameters.
    Shows how map complexity affects learning performance.
    """
    print("\nRunning Experiment 1: Map Complexity...")
    results = []
    trials = [(map_name, path, agent_type)
              for map_name, path in MAP_PATHS
              for agent_type in ["SARSA", "QLearn"]]

    for map_name, path, agent_type in tqdm(trials, desc="Exp 1", unit="trial"):
        grid = mc.compress(path)
        target = find_target(grid)
        r = run_trial(grid, target, agent_type, episodes, steps, epsilon, discount, strategy)
        r["map"] = map_name
        results.append(r)

    print_table(results,
                ["map", "agent", "time_s", "episodes", "accuracy", "avg_path"],
                f"Exp 1 – Map Complexity  [eps={epsilon}, gamma={discount}, {strategy}]")
    return results


def exp2_exploration_rate(episodes=EPISODES, steps=STEPS,
                           discount=0.5, strategy="S1"):
    """
    Experiment 2: Exploration Rate
    Run both agents on Map 4 with gamma=0.5 and epsilon in {0, 0.5, 1}.
    Shows the tradeoff between pure exploitation (eps=0) and pure exploration (eps=1).
    """
    print("\nRunning Experiment 2: Exploration Rate (Map 4)...")
    grid = mc.compress(mc.MAP_4_PATH)
    target = find_target(grid)
    results = []
    trials = [(eps, agent_type)
              for eps in [0, 0.5, 1]
              for agent_type in ["SARSA", "QLearn"]]

    for epsilon, agent_type in tqdm(trials, desc="Exp 2", unit="trial"):
        r = run_trial(grid, target, agent_type, episodes, steps, epsilon, discount, strategy)
        results.append(r)

    print_table(results,
                ["agent", "epsilon", "discount", "time_s", "episodes", "accuracy", "avg_path"],
                f"Exp 2 – Exploration Rate  [gamma={discount}, {strategy}]")
    return results


def exp3_discount_value(episodes=EPISODES, steps=STEPS,
                         epsilon=0.5, strategy="S1"):
    """
    Experiment 3: Discount Value
    Run both agents on Map 4 with epsilon=0.5 and gamma in {0.1, 0.5, 1}.
    Shows the effect of short-sighted (low gamma) vs long-sighted (high gamma) planning.
    """
    print("\nRunning Experiment 3: Discount Value (Map 4)...")
    grid = mc.compress(mc.MAP_4_PATH)
    target = find_target(grid)
    results = []
    trials = [(disc, agent_type)
              for disc in [0.1, 0.5, 1]
              for agent_type in ["SARSA", "QLearn"]]

    for discount, agent_type in tqdm(trials, desc="Exp 3", unit="trial"):
        r = run_trial(grid, target, agent_type, episodes, steps, epsilon, discount, strategy)
        results.append(r)

    print_table(results,
                ["agent", "epsilon", "discount", "time_s", "episodes", "accuracy", "avg_path"],
                f"Exp 3 – Discount Value  [eps={epsilon}, {strategy}]")
    return results


def exp4_reward_strategy(best_epsilon, best_discount, episodes=EPISODES, steps=STEPS,
                          strategies=["S1", "S2", "S3"]):
    """
    Experiment 4: Reward Strategy
    Run both agents on Map 4 with the best epsilon and discount from experiments 2 and 3.
    Compare performance across reward strategies S1, S2, and S3.
    """
    print(f"\nRunning Experiment 4: Reward Strategy (Map 4, eps={best_epsilon}, gamma={best_discount})...")
    grid = mc.compress(mc.MAP_4_PATH)
    target = find_target(grid)
    results = []
    trials = [(s, agent_type)
              for s in strategies
              for agent_type in ["SARSA", "QLearn"]]

    for strategy, agent_type in tqdm(trials, desc="Exp 4", unit="trial"):
        r = run_trial(grid, target, agent_type, episodes, steps, best_epsilon, best_discount, strategy)
        results.append(r)

    print_table(results,
                ["agent", "strategy", "epsilon", "discount", "time_s", "episodes", "accuracy", "avg_path"],
                f"Exp 4 – Reward Strategy  [eps={best_epsilon}, gamma={best_discount}]")
    return results


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # np.random.seed(SEED)

    r1 = exp1_map_complexity()
    r2 = exp2_exploration_rate()
    r3 = exp3_discount_value()

    # Pick best epsilon from exp2 and best discount from exp3
    best_eps = best_result(r2)["epsilon"]
    best_disc = best_result(r3)["discount"]

    print(f"\nBest epsilon (from Exp 2): {best_eps}")
    print(f"Best discount (from Exp 3): {best_disc}")

    r4 = exp4_reward_strategy(best_eps, best_disc)
