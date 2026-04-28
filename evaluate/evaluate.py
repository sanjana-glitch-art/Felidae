"""
evaluate.py — Load a trained CatRL model and run evaluation episodes.

Run from the catrl/ root directory:
    python evaluate/evaluate.py

Options (edit at the top of this file):
    MODEL_PATH  — path to the saved .zip model
    N_EPISODES  — how many evaluation episodes to run
    RENDER      — True = show pygame window, False = headless (faster)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict
from stable_baselines3 import PPO
from env.cat_env import CatEnv

# ──────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────
MODEL_PATH = "models/catrl_v2"   # no .zip extension needed
N_EPISODES = 10
RENDER     = True                # set False for headless evaluation

# ──────────────────────────────────────────────────────────────
#  ACTION LABELS  (for personality summary)
# ──────────────────────────────────────────────────────────────
ACTION_NAMES = {
    0:  "move_up",
    1:  "move_down",
    2:  "move_left",
    3:  "move_right",
    4:  "zoom",
    5:  "nap",
    6:  "eat",
    7:  "hunt_toy",
    8:  "knock_object",
    9:  "sit_in_box",
    10: "meow",
    11: "seek_human",
    12: "ignore_human",
    13: "stare_at_wall",
}


def run_evaluation():
    # ── load model ────────────────────────────────────────────
    print(f"\nLoading model from {MODEL_PATH}.zip ...")
    model = PPO.load(MODEL_PATH)

    # ── create env ────────────────────────────────────────────
    render_mode = "human" if RENDER else None
    env = CatEnv(render_mode=render_mode)

    # ── tracking ──────────────────────────────────────────────
    episode_rewards  = []
    episode_lengths  = []
    action_counts    = defaultdict(int)
    visit_counts     = np.zeros((env.GRID_H, env.GRID_W), dtype=np.int32)
    termination_log  = []   # "vet" or "truncated"

    print(f"\nRunning {N_EPISODES} evaluation episodes...\n")

    for ep in range(1, N_EPISODES + 1):
        obs, _    = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            # model.predict returns (action, state)
            # deterministic=True = always take the best known action (no exploration)
            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            done       = terminated or truncated

            # record what the cat did
            action_counts[action] += 1

            # record where the cat was
            r, c = info["cat_pos"]
            visit_counts[r, c] += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        termination_log.append("vet" if terminated else "truncated")

        print(f"  Episode {ep:2d} | reward: {ep_reward:7.1f} | "
              f"steps: {ep_steps:4d} | ended: {termination_log[-1]}")

    env.close()

    # ──────────────────────────────────────────────────────────
    #  PERSONALITY SUMMARY
    # ──────────────────────────────────────────────────────────
    total_actions = sum(action_counts.values())
    print(f"\n{'='*55}")
    print(f"  EVALUATION SUMMARY  ({N_EPISODES} episodes)")
    print(f"{'='*55}")
    print(f"  Mean reward   : {np.mean(episode_rewards):7.1f}")
    print(f"  Std reward    : {np.std(episode_rewards):7.1f}")
    print(f"  Min / Max     : {np.min(episode_rewards):.1f} / "
          f"{np.max(episode_rewards):.1f}")
    print(f"  Mean ep len   : {np.mean(episode_lengths):.0f} steps")
    print(f"  Vet endings   : {termination_log.count('vet')} / {N_EPISODES}")
    print()
    print("  ACTION BREAKDOWN  (% of all steps):")
    for action_id in sorted(action_counts.keys(),
                             key=lambda a: action_counts[a], reverse=True):
        pct  = action_counts[action_id] / total_actions * 100
        name = ACTION_NAMES[action_id]
        bar  = "█" * int(pct / 2)
        print(f"    {name:<15} {pct:5.1f}%  {bar}")

    print()

    # ── derive personality label ───────────────────────────────
    nap_pct      = action_counts[5]  / total_actions * 100
    knock_pct    = action_counts[8]  / total_actions * 100
    box_pct      = action_counts[9]  / total_actions * 100
    seek_pct     = action_counts[11] / total_actions * 100
    ignore_pct   = action_counts[12] / total_actions * 100
    wall_pct     = action_counts[13] / total_actions * 100

    print("  CAT PERSONALITY PROFILE:")
    if nap_pct > 25:
        print("    😴  This cat is a PROFESSIONAL SLEEPER. It naps constantly.")
    if knock_pct > 15:
        print("    💥  This cat is a CHAOS AGENT. Knocking things is its purpose.")
    if box_pct > 10:
        print("    📦  This cat has a BOX OBSESSION. Cardboard is home.")
    if seek_pct > ignore_pct:
        print("    🐱  This cat is SOCIAL. It likes humans (sometimes).")
    else:
        print("    😤  This cat is ALOOF. Humans are beneath it.")
    if wall_pct > 5:
        print("    🧱  This cat stares at walls. Existential.")

    # ──────────────────────────────────────────────────────────
    #  SAVE VISIT COUNTS  (used by viz/heatmap.py)
    # ──────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    np.save("logs/visit_counts.npy", visit_counts)
    print(f"\n  Visit heatmap saved to logs/visit_counts.npy")
    print(f"  Run  python viz/heatmap.py  to visualise it.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_evaluation()