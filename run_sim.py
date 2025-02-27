# This script is just to run through the motions simulating with mjx

import jax
from air_hockey_challenge.framework import (
    AirHockeyChallengeGymWrapper,
)

import numpy as np

env = AirHockeyChallengeGymWrapper("3dof-hit", interpolation_order=2)

env.reset()

n_envs = 2

for i in range(1000):

    action = env.action_space.sample()

    # Create jax array, and replicate it to n_envs
    action = jax.numpy.array(action)
    # Stack jax array n_envs times
    action = jax.numpy.stack([action] * n_envs)

    # Convert to numpy array
    action = np.array(action)

    obs, reward, done, info = jax.jit(env.step)(action)
    # env.render()
    if done:
        env.reset()
