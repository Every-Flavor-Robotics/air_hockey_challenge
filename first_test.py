# This script is just to run through the motions of training an agent for the air hockey challenge

import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import wandb
from air_hockey_challenge.framework import AirHockeyChallengeGymWrapper
from wandb.integration.sb3 import WandbCallback


def get_env():
    return AirHockeyChallengeGymWrapper("3dof-hit", interpolation_order=2)


model_path = "ppo_air_hockey"

model = PPO.load(model_path)

vec_env = make_vec_env(get_env, n_envs=1)

obs = vec_env.reset()
step_count = 0
max_steps = 800

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

    step_count += 1
    if step_count >= max_steps:
        obs = vec_env.reset()
        step_count = 0

    # Display image
    # cv2.imshow("Air Hockey", image)

    # # Sleep for 10ms, or until a key is pressed
    # if cv2.waitKey(10) != -1:
    #     break
