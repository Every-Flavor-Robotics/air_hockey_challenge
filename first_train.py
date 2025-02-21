# This script is just to run through the motions of training an agent for the air hockey challenge

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from air_hockey_challenge.framework import AirHockeyChallengeGymWrapper


def get_env():
    return AirHockeyChallengeGymWrapper("3dof-hit", interpolation_order=2)


# Parallel environments
vec_env = make_vec_env(get_env, n_envs=12)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model.learn(total_timesteps=250000)
