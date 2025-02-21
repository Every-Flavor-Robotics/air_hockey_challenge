# This script is just to run through the motions of training an agent for the air hockey challenge

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import wandb
from air_hockey_challenge.framework import AirHockeyChallengeGymWrapper
from wandb.integration.sb3 import WandbCallback


def get_env():
    return AirHockeyChallengeGymWrapper("3dof-hit", interpolation_order=2)


wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="air-hockey",
    # Track hyperparameters and run metadata
    config={},
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optiona
)


# Parallel environments
vec_env = make_vec_env(get_env, n_envs=16)

model = PPO(
    "MlpPolicy", vec_env, verbose=1, device="cpu", tensorboard_log=f"runs/{run.id}"
)
model.learn(
    total_timesteps=2500000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)


run.finish()


# Save the model
model.save("ppo_air_hockey")
