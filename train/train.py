"""
train.py — Train the CatRL agent using PPO.

Run from the catrl/ root directory:
    python train/train.py

To monitor training live, open a SECOND terminal and run:
    tensorboard --logdir ./logs/
Then open http://localhost:6006 in your browser.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env.cat_env import CatEnv


# ──────────────────────────────────────────────────────────────
#  CONFIG  — edit these to try different experiments
# ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "catrl_v2"          # change to v2, v3 for ablations
TOTAL_TIMESTEPS = 500_000             # increase to 1_000_000 for better behaviour
N_ENVS          = 4                   # parallel environments (speeds training up)
SAVE_FREQ       = 50_000              # save a checkpoint every N steps

# ──────────────────────────────────────────────────────────────
#  MAKE DIRECTORIES
# ──────────────────────────────────────────────────────────────
os.makedirs("models",                    exist_ok=True)
os.makedirs(f"logs/{EXPERIMENT_NAME}",   exist_ok=True)
os.makedirs("models/checkpoints",        exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  CREATE VECTORISED ENVIRONMENT
# ──────────────────────────────────────────────────────────────
# make_vec_env runs N_ENVS copies of CatEnv in parallel.
# PPO collects experience from all of them simultaneously — this
# is why vectorised envs speed up training significantly.
env = make_vec_env(CatEnv, n_envs=N_ENVS)


# ──────────────────────────────────────────────────────────────
#  SEPARATE EVAL ENV  (single env, no parallelism needed)
# ──────────────────────────────────────────────────────────────
eval_env = make_vec_env(CatEnv, n_envs=1)


# ──────────────────────────────────────────────────────────────
#  CALLBACKS
# ──────────────────────────────────────────────────────────────
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"models/{EXPERIMENT_NAME}_best",
    log_path=f"logs/{EXPERIMENT_NAME}/eval",
    eval_freq=10_000,        # evaluate every 10k steps
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ // N_ENVS,   # divided by N_ENVS because of parallelism
    save_path="models/checkpoints",
    name_prefix=EXPERIMENT_NAME,
    verbose=1,
)


# ──────────────────────────────────────────────────────────────
#  BUILD THE PPO MODEL
# ──────────────────────────────────────────────────────────────
#
#  "MlpPolicy" = multi-layer perceptron (two hidden layers of 64 neurons each).
#  This is the right choice for flat observation vectors like ours (12 numbers).
#  If your observations were images you'd use "CnnPolicy" instead.
#
#  Key hyperparameters worth tweaking:
#    learning_rate — how fast the policy updates. Lower = more stable, slower.
#    n_steps       — how many steps per env to collect before updating. Bigger = more stable.
#    batch_size    — mini-batch size for gradient updates. Must divide n_steps * n_envs.
#    ent_coef      — entropy bonus. Higher = more exploration, more random behaviour early on.
#
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=f"./logs/{EXPERIMENT_NAME}",
    learning_rate=3e-4,      # default, sensible starting point
    n_steps=2048,            # steps collected per env per update
    batch_size=64,
    n_epochs=10,             # how many gradient steps per batch
    gamma=0.99,              # discount factor — how much future rewards matter
    gae_lambda=0.95,         # GAE smoothing factor
    ent_coef=0.01,           # encourages exploration early on
    clip_range=0.2,          # the "proximal" part of PPO — limits policy change per step
    policy_kwargs=dict(net_arch=[128, 128]),  # two hidden layers of 128 neurons
)

print(f"\n{'='*50}")
print(f"  Training {EXPERIMENT_NAME}")
print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
print(f"  Parallel envs   : {N_ENVS}")
print(f"  TensorBoard     : tensorboard --logdir ./logs/{EXPERIMENT_NAME}")
print(f"{'='*50}\n")


# ──────────────────────────────────────────────────────────────
#  TRAIN
# ──────────────────────────────────────────────────────────────
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True,
)


# ──────────────────────────────────────────────────────────────
#  SAVE FINAL MODEL
# ──────────────────────────────────────────────────────────────
save_path = f"models/{EXPERIMENT_NAME}"
model.save(save_path)
print(f"\nModel saved to {save_path}.zip")


# ──────────────────────────────────────────────────────────────
#  REWARD ABLATION REMINDER
# ──────────────────────────────────────────────────────────────
print("""
──────────────────────────────────────────────────────────────
  NEXT EXPERIMENTS (edit cat_env.py reward values, then retrain):
  
  v2 — Double the nap reward (+20 on sunny spot). Does the cat
       become a professional sleeper?

  v3 — Remove the time penalty (-0.5 per step). Does the cat
       get stuck doing nothing?

  v4 — Double the knock_object reward (+14). Does the cat
       become a chaos agent?
──────────────────────────────────────────────────────────────
""")